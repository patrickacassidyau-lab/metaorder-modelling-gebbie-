"""Tick-level metaorder detection signal (equations state, entry, sizing, algorithm)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metaorder_signal.powerlaw_fit import (
    expected_total_length_given_partial,
    survival_continuous_approximation,
)


@dataclass
class MetaorderSignalParams:
    """Structural estimates (no return optimisation)."""

    alpha: float
    gamma1: float
    gamma2: float
    beta: float
    sigma_d: float
    vd: float


@dataclass
class SignalConfig:
    """Entry / exit defaults from the specification."""

    p_min: float = 0.6
    phi_entry: float = 0.4
    rho_max: float = 1.5
    n_min: int = 3
    phi_exit: float = 0.8
    s_max: float = 1.0
    kappa_cost: float = 0.01
    unreliable_gamma2: float = 0.8


def process_trade_stream(
    trades: pd.DataFrame,
    params: MetaorderSignalParams,
    cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    """
    Event-ordered computation of survival, phi, rho, and discrete entry/exit events.

    Required columns: mid (float), quantity (float >0), sign (+1/-1), timestamp.

    Theoretical impact (algorithm / equation concave):
        I_theo = gamma1 * sigma_d * sqrt(Q_hat) * phi**gamma2
    Observed impact:
        I_obs = eps_run * (ln m - ln m0_run)
    """
    cfg = cfg or SignalConfig()
    df = trades.sort_values("timestamp").reset_index(drop=True)
    mids = df["mid"].to_numpy(dtype=float)
    qty = df["quantity"].to_numpy(dtype=float)
    sgn = np.sign(df["sign"].to_numpy(dtype=float)).astype(int)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype(np.int64) / 1e9

    n = len(df)
    survival_p = np.full(n, np.nan)
    phi = np.full(n, np.nan)
    rho = np.full(n, np.nan)
    i_obs = np.full(n, np.nan)
    i_theo = np.full(n, np.nan)
    q_hat_arr = np.full(n, np.nan)
    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    exit_signal = np.zeros(n, dtype=bool)
    size = np.zeros(n, dtype=float)

    eps_run = 0
    n_run = 0
    q_run = 0.0
    m0_run = np.nan
    t0_run = np.nan

    position = 0.0
    prev_trade_sign = 0

    for i in range(n):
        if not np.isfinite(mids[i]) or mids[i] <= 0 or qty[i] <= 0 or sgn[i] == 0:
            prev_trade_sign = sgn[i]
            continue

        run_broken = prev_trade_sign != 0 and sgn[i] != prev_trade_sign

        if eps_run == 0:
            eps_run = int(sgn[i])
            n_run = 1
            q_run = float(qty[i])
            m0_run = float(mids[i])
            t0_run = float(ts[i])
        elif int(sgn[i]) == eps_run:
            n_run += 1
            q_run += float(qty[i])
        else:
            eps_run = int(sgn[i])
            n_run = 1
            q_run = float(qty[i])
            m0_run = float(mids[i])
            t0_run = float(ts[i])

        mean_child = q_run / max(n_run, 1)
        l_tot = expected_total_length_given_partial(n_run, params.alpha)
        q_hat = float(l_tot * mean_child)
        q_hat_arr[i] = q_hat

        surv = survival_continuous_approximation(n_run, params.alpha)
        survival_p[i] = surv

        ph = float(q_run / max(q_hat, 1e-12))
        ph = float(min(ph, 1.0))
        phi[i] = ph

        i_o = float(eps_run) * (np.log(mids[i]) - np.log(m0_run))
        i_obs[i] = i_o

        i_t = float(
            params.gamma1
            * params.sigma_d
            * np.sqrt(max(q_hat, 0.0))
            * (ph**params.gamma2)
        )
        i_theo[i] = i_t

        rho[i] = float(i_o / i_t) if np.isfinite(i_t) and abs(i_t) > 1e-18 else np.nan

        if position != 0.0:
            if run_broken:
                exit_signal[i] = True
                position = 0.0
            elif ph > cfg.phi_exit:
                exit_signal[i] = True
                position = 0.0

        entry_ok = (
            surv > cfg.p_min
            and ph < cfg.phi_entry
            and np.isfinite(rho[i])
            and rho[i] < cfg.rho_max
            and n_run >= cfg.n_min
            and params.gamma2 < cfg.unreliable_gamma2
        )

        if position == 0.0 and entry_ok and not exit_signal[i]:
            sz = cfg.s_max * surv * (1.0 - ph)
            position = sz * eps_run
            if eps_run > 0:
                entry_long[i] = True
            else:
                entry_short[i] = True
            size[i] = abs(sz)

        prev_trade_sign = int(sgn[i])

    out = df.copy()
    out["survival_p"] = survival_p
    out["phi"] = phi
    out["rho"] = rho
    out["i_obs"] = i_obs
    out["i_theo"] = i_theo
    out["q_hat"] = q_hat_arr
    out["entry_long"] = entry_long
    out["entry_short"] = entry_short
    out["exit_signal"] = exit_signal
    out["signal_size"] = size
    return out
