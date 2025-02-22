from functools import partial
from typing import Tuple

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from pydantic import BaseModel, Field, PrivateAttr
import copy
from basicpy.tools.dct_tools import JaxDCT

idct2d, dct2d, idct3d, dct3d = JaxDCT.idct2d, JaxDCT.dct2d, JaxDCT.idct3d, JaxDCT.dct3d
newax = jnp.newaxis


@jit
def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


class BaseFit(BaseModel):
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    max_mu: float = Field(0, description="The maximum value of mu.")
    init_mu: float = Field(0, description="Initial value for mu.")
    D_Z_max: float = Field(0, description="Maximum value for D_Z.")
    image_norm: float = Field(0, description="The 2nd order norm for the images.")
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-6,
        description="Optimization tolerance for update diff.",
    )
    smoothness_darkfield: float = Field(
        0.0,
        description="Darkfield smoothness weight for sparse reguralization.",
    )
    sparse_cost_darkfield: float = Field(
        0.0,
        description="Darkfield sparseness weight for sparse reguralization.",
    )
    smoothness_flatfield: float = Field(
        0.0,
        description="Flatfield smoothness weight for sparse reguralization.",
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )

    class Config:
        frozen = True
        extra = "ignore"

    def _cond(self, vals):
        k = vals[0]

        fit_residual = vals[-2]
        value_diff = vals[-1]
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm

        conv = jnp.any(
            jnp.array(
                [
                    norm_ratio > self.optimization_tol,
                    norm_ratio > self.optimization_tol,
                ]
            )
        )
        return jnp.all(
            jnp.array(
                [
                    conv,
                    k < self.max_iterations,
                ]
            )
        )

    @jit
    def _fit_jit(
        self,
        Im,
        W,
        W_D,
        S,
        S_hat,
        D_R,
        D_Z,
        B,
        I_B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf
        value_diff = jnp.inf

        vals = (0, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step,
            Im,
            W,
            W_D,
        )
        #        while self._cond(vals):
        #            vals = step(vals)
        vals = lax.while_loop(self._cond, step, vals)

        k, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return S, S_hat, D_R, D_Z, I_B, I_R, B, norm_ratio, k < self.max_iterations

    @jit
    def _fit_baseline_jit(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf
        value_diff = jnp.inf

        vals = (0, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step_only_baseline,
            Im,
            W,
            S,
            D,
        )

        #        while self._cond(vals):
        #            vals = step(vals)
        vals = lax.while_loop(self._cond, step, vals)
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return I_R, B, norm_ratio, k < self.max_iterations

    def fit(
        self,
        Im,
        W,
        W_D,
        S,
        S_hat,
        D_R,
        D_Z,
        B,
        I_B,
        I_R,
    ):
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D_R.shape != Im.shape[1:]:
            raise ValueError("D_R must have the same shape as images.shape[1:]")
        if not jnp.isscalar(D_Z):
            raise ValueError("D_Z must be a scalar.")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        if W_D.shape != Im.shape[1:]:
            raise ValueError(
                "darkfield weight must have the same shape as images.shape[1:]"
            )
        return self._fit_jit(Im, W, W_D, S, S_hat, D_R, D_Z, B, I_B, I_R)

    def fit_baseline(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool]:
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D.shape != Im.shape[1:]:
            raise ValueError("D must have the same shape as images.shape[1:]")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        return self._fit_baseline_jit(Im, W, S, D, B, I_R)

    def tree_flatten(self):
        # all of the fields are treated as "static" values for JAX
        children = []
        aux_data = self.dict()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, _children):
        return cls(**aux_data)


@register_pytree_node_class
class LadmapFit(BaseFit):
    @jit
    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        # k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals
        k, S, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D_R[newax, ...] + D_Z
        eta_S = jnp.sum(B**2) * 1.02 + 0.01
        S_new = (
            S
            + jnp.sum(B[:, newax, newax, newax] * (Im - I_B - I_R + Y / mu), axis=0)
            / eta_S
        )
        S_new = idct3d(
            _jshrinkage(dct3d(S_new), self.smoothness_flatfield / (eta_S * mu))
        )
        S_new = jnp.where(S_new.min() < 0, S_new - S_new.min(), S_new)
        dS = S_new - S
        S = S_new

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D_R[newax, ...] + D_Z
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        S_sq = jnp.sum(S**2)
        B_new = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2, 3)) / S_sq
        B_new = jnp.where(S_sq > 0, B_new, B)
        B_new = jnp.maximum(B_new, 0)

        mean_B = jnp.mean(B_new)
        B_new = jnp.where(mean_B > 0, B_new / mean_B, B_new)
        S = jnp.where(mean_B > 0, S * mean_B, S)

        dB = B_new - B
        B = B_new

        BS = S[newax, ...] * B[:, newax, newax, newax]
        if self.get_darkfield:
            D_Z_new = jnp.mean(Im - BS - D_R[newax, ...] - I_R + Y / 2.0 / mu)
            D_Z_new = jnp.clip(D_Z_new, 0, self.D_Z_max)
            dD_Z = D_Z_new - D_Z
            D_Z = D_Z_new

            eta_D = Im.shape[0] * 1.02
            D_R_new = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[newax, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R_new = idct3d(
                _jshrinkage(dct3d(D_R_new), self.smoothness_darkfield / eta_D / mu)
            )
            D_R_new = _jshrinkage(
                D_R_new, self.sparse_cost_darkfield * dark_weight / eta_D / mu
            )
            dD_R = D_R_new - D_R
            D_R = D_R_new

        I_B = BS + D_R[newax, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = jnp.max(
            jnp.array(
                [
                    jnp.linalg.norm(dS.ravel(), ord=2) * jnp.sqrt(eta_S),
                    jnp.linalg.norm(dI_R.ravel(), ord=2) * jnp.sqrt(1.0),
                    # TODO find better form with theoretical evidence
                    jnp.linalg.norm(dB.ravel(), ord=2),
                ]
            )
        )

        if self.get_darkfield:
            value_diff = jnp.max(
                jnp.array(
                    [
                        value_diff,
                        jnp.linalg.norm(dD_R.ravel(), ord=2) * jnp.sqrt(eta_D),
                        # TODO find better form with theoretical evidence
                        dD_Z**2,
                    ]
                )
            )
        value_diff = value_diff / self.image_norm
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff)

    @jit
    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D[newax, ...]
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        B_new = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2, 3)) / jnp.sum(S**2)
        B_new = jnp.maximum(B_new, 0)
        dB = B_new - B
        B = B_new

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D[newax, ...]
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = jnp.max(
            jnp.array(
                [
                    jnp.linalg.norm(dI_R.ravel(), ord=2) * jnp.sqrt(1.0),
                    # TODO find better form with theoretical evidence
                    jnp.linalg.norm(dB.ravel(), ord=2),
                ]
            )
        )
        value_diff = value_diff / self.image_norm

        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, I_R, B, Y, mu, fit_residual, value_diff)

    def calc_weights(self, I_B, I_R):
        Ws = jnp.ones_like(I_R, dtype=jnp.float32) / (
            jnp.abs(I_R / (I_B + self.epsilon)) + self.epsilon
        )
        return Ws / jnp.mean(Ws)

    def calc_dark_weights(self, D_R):
        Ws = np.ones_like(D_R, dtype=jnp.float32) / (jnp.abs(D_R) + self.epsilon)
        return Ws / jnp.mean(Ws)

    def calc_weights_baseline(self, I_B, I_R):
        return self.calc_weights(I_B, I_R)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z


@register_pytree_node_class
class ApproximateFit(BaseFit):
    _ent1: float = PrivateAttr(1.0)
    _ent2: float = PrivateAttr(10.0)

    # @jit
    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        # approximate fitting only accepts two-dimensional images.

        k, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, _ = vals

        s_s, _, s_m, s_n = Im.shape
        Im = Im[:, 0, ...].reshape(s_s, -1)
        weight = weight[:, 0, ...].reshape(s_s, -1)
        D_R = D_R[0, ...]
        I_R = I_R[:, 0, ...].reshape(s_s, -1)
        Y = Y[:, 0, ...].reshape(s_s, -1)

        S = idct2d(S_hat)

        I_B = S * B[:, newax, newax] + D_R[newax, ...]
        I_B = I_B.reshape(s_s, -1)

        temp_W = (Im - I_R - I_B + Y / mu) / self._ent1
        temp_W = jnp.mean(temp_W, axis=0)
        S_hat = S_hat + dct2d(temp_W.reshape(s_m, s_n))
        S_hat = _jshrinkage(S_hat, self.smoothness_flatfield / (self._ent1 * mu))
        S = idct2d(S_hat)

        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...]
        I_B = I_B.reshape(s_s, -1)

        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))

        R = Im - I_R
        B = jnp.mean(R, axis=1) / jnp.mean(R)
        B = jnp.maximum(B, 0)

        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...]
        I_B = I_B.reshape(s_s, -1)

        """
        if self.get_darkfield:

            S_inmask = S.reshape(-1) >= jnp.mean(S)
            S_outmask = S.reshape(-1) < jnp.mean(S)

            R_0 = copy.deepcopy(R)
            R_1 = copy.deepcopy(R)

            R_0 = jnp.where(S_inmask[newax, ...], R_0, jnp.nan)
            R_1 = jnp.where(S_outmask[newax, ...], R_1, jnp.nan)

            B1_coeff = (jnp.nanmean(R_0, 1) - jnp.nanmean(R_1, 1)) / (
                jnp.mean(R) + 1e-6
            )

            k = len(B)

            temp1 = jnp.nansum(B**2)
            temp2 = jnp.nansum(B)
            temp3 = jnp.nansum(B1_coeff)
            temp4 = jnp.nansum(B * B1_coeff)
            temp5 = temp2 * temp3 - k * temp4

            D_Z = jnp.where(temp5 == 0, 0, (temp1 * temp3 - temp2 * temp4) / temp5)
            D_Z = jnp.maximum(D_Z, 0)
            D_Z = jnp.minimum(D_Z, Im.min() / (jnp.mean(S) + 1e-6))

            Z = D_Z * jnp.mean(S) - D_Z * S.reshape(-1)

            A1_offset = jnp.nanmean(R, 0) - jnp.nanmean(B) * S.reshape(-1)
            # A1_offset = A1_offset - jnp.mean(A1_offset)
            D_R = A1_offset - jnp.mean(A1_offset) - Z

            D_R = dct2d(D_R.reshape(mm, nn))
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))

            D_R = D_R + Z.reshape(mm, nn)
        """
        """
        if self.get_darkfield:

            validA1coeff_idx = B < 1

            S_inmask = S.reshape(-1) >= jnp.mean(S)
            S_outmask = S.reshape(-1) < jnp.mean(S)

            R_0 = copy.deepcopy(R)
            R_1 = copy.deepcopy(R)

            R_0 = jnp.where(
                S_inmask[newax, ...] * validA1coeff_idx[..., newax], R_0, jnp.nan
            )
            R_1 = jnp.where(
                S_outmask[newax, ...] * validA1coeff_idx[..., newax], R_1, jnp.nan
            )

            B1_coeff = (jnp.nanmean(R_0, 1) - jnp.nanmean(R_1, 1)) / (
                jnp.mean(R) + 1e-6
            )

            k = jnp.sum(validA1coeff_idx)

            B_nan = jnp.where(validA1coeff_idx, B, jnp.nan)

            temp1 = jnp.nansum(B_nan**2)
            temp2 = jnp.nansum(B_nan)
            temp3 = jnp.nansum(B1_coeff)
            temp4 = jnp.nansum(B_nan * B1_coeff)
            temp1 = jnp.nan_to_num(temp1)
            temp2 = jnp.nan_to_num(temp2)
            temp3 = jnp.nan_to_num(temp3)
            temp4 = jnp.nan_to_num(temp4)
            temp5 = temp2 * temp3 - k * temp4

            D_Z = jnp.where(temp5 == 0, 0, (temp1 * temp3 - temp2 * temp4) / temp5)
            D_Z = jnp.maximum(D_Z, 0)
            D_Z = jnp.minimum(D_Z, Im.min())

            Z = D_Z * jnp.mean(S) - D_Z * S.reshape(-1)

            R_nan = jnp.where(validA1coeff_idx[:, None], R, jnp.nan)

            A1_offset = jnp.nanmean(R_nan, 0) - jnp.nanmean(B_nan) * S.reshape(-1)  # Z
            # A1_offset = jnp.nan_to_num(A1_offset)
            A1_offset = A1_offset - jnp.nanmean(A1_offset)
            D_R = A1_offset - D_Z * (jnp.mean(S) - S.reshape(-1))

            D_R = dct2d(D_R.reshape(s_m, s_n))
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))

            D_R = D_R + D_Z
        """

        if self.get_darkfield:

            validA1coeff_idx = B < 1

            S_inmask = S.reshape(-1) >= jnp.mean(S)
            S_outmask = S.reshape(-1) < jnp.mean(S)

            R_0 = copy.deepcopy(R)
            R_1 = copy.deepcopy(R)

            R_0 = jnp.where(
                S_inmask[newax, ...] * validA1coeff_idx[..., newax], R_0, jnp.nan
            )
            R_1 = jnp.where(
                S_outmask[newax, ...] * validA1coeff_idx[..., newax], R_1, jnp.nan
            )

            B1_coeff = (jnp.nanmean(R_0, 1) - jnp.nanmean(R_1, 1)) / (
                jnp.mean(R) + 1e-6
            )

            k = jnp.sum(validA1coeff_idx)

            B_nan = jnp.where(validA1coeff_idx, B, jnp.nan)

            temp1 = jnp.nansum(B_nan**2)
            temp2 = jnp.nansum(B_nan)
            temp3 = jnp.nansum(B1_coeff)
            temp4 = jnp.nansum(B_nan * B1_coeff)
            temp1 = jnp.nan_to_num(temp1)
            temp2 = jnp.nan_to_num(temp2)
            temp3 = jnp.nan_to_num(temp3)
            temp4 = jnp.nan_to_num(temp4)
            temp5 = temp2 * temp3 - k * temp4

            D_Z = jnp.where(temp5 == 0, 0, (temp1 * temp3 - temp2 * temp4) / temp5)
            D_Z = jnp.maximum(D_Z, 0)
            D_Z = jnp.minimum(D_Z, Im.min() / jnp.mean(S))

            Z = D_Z * jnp.mean(S) - D_Z * S.reshape(-1)

            R_nan = jnp.where(validA1coeff_idx[:, None], R, jnp.nan)

            A1_offset = (
                jnp.nanmean(R_nan, 0)
                - jnp.nanmean(B_nan[..., newax]) * S.reshape(-1)[newax, ...]
            )
            A1_offset = A1_offset - jnp.nanmean(A1_offset)
            # D_R = A1_offset - jnp.nanmean(A1_offset) - Z
            D_R = A1_offset - jnp.mean(A1_offset) - Z

            D_R = dct2d(D_R.reshape(s_m, s_n))
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))

            D_R = D_R + Z.reshape(s_m, s_n)

        fit_residual = Im - I_B - I_R
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        # put the variables back to 4-dim input array
        S = S[newax, ...]
        D_R = D_R[newax, ...]
        I_R = I_R[:, newax, ...]
        I_B = I_B[:, newax, ...]
        Y = Y[:, newax, ...]
        fit_residual = fit_residual[:, newax, ...]
        return (
            k + 1,
            S,
            S_hat,
            D_R,
            D_Z,
            I_B.reshape(s_s, 1, s_m, s_n),
            I_R.reshape(s_s, 1, s_m, s_n),
            B,
            Y.reshape(s_s, 1, s_m, s_n),
            mu,
            fit_residual.reshape(s_s, 1, s_m, s_n),
            0.0,
        )

    @jit
    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        Im = Im[:, 0, ...]
        weight = weight[:, 0, ...]
        S = S[0]
        D = D[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        I_B = S[newax, ...] * B[:, newax, newax] + D[newax, ...]

        # update I_R using approximated l0 norm
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))

        R1 = Im - I_R
        # A1_coeff = mean(R1)-mean(A_offset);
        B = jnp.mean(R1, axis=(1, 2)) - jnp.mean(D)
        # A1_coeff(A1_coeff<0) = 0;
        B = jnp.maximum(B, 0)
        # Z1 = D - A1_hat - E1_hat;
        fit_residual = Im - I_B - I_R
        # Y1 = Y1 + mu*Z1;
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        I_R = I_R[:, newax, ...]
        Y = Y[:, newax, ...]
        fit_residual = fit_residual[:, newax, ...]
        return (k + 1, I_R, B, Y, mu, fit_residual, 0.0)

    def calc_weights(
        self,
        I_B,
        I_R,
        Ws2,
        epsilon,
    ):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        XE_norm = I_R / (jnp.mean(I_B, axis=(1, 2))[:, newax, newax] + 1e-6)
        weight = jnp.ones_like(I_R) / (jnp.abs(XE_norm) + self.epsilon)

        weight = jnp.where(
            Ws2[:, 0, ...] == 0,
            weight * epsilon,
            weight,
        )
        weight = weight * weight.size / weight.sum()

        return weight[:, newax, ...]

    def calc_dark_weights(self, D_R):
        return jnp.ones_like(D_R)

    def calc_weights_baseline(self, I_B, I_R):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        mean_vec = jnp.mean(I_B, axis=(1, 2))
        XE_norm = mean_vec[:, newax, newax] / (I_R + 1e-6)
        weight = 1.0 / (jnp.abs(XE_norm) + self.epsilon)
        weight = weight / jnp.mean(weight)
        return weight[:, newax, ...]

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R
