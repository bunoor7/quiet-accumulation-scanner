"""Human-readable output, in the same register as the methodology this
project mirrors: direct, concise journalistic Arabic, no dashes, no
colons inside sentences. Two shapes only:

- No signal this run -> one short line and nothing else.
- A real, data-backed signal -> the full seven-point report.

Nothing in between. A "maybe" is reported as no signal.
"""
from __future__ import annotations

from typing import List, Optional

from .market_timing import MarketTimingResult, MarketRegime
from .multi_timeframe import ConfirmedCandidate
from .signal_engine import SignalVerdict
from .trade_plan import TradePlan

_REGIME_AR = {
    MarketRegime.SHARP_TREND: "حركة اتجاهية حادة على مستوى السوق ككل",
    MarketRegime.ACTIVE_RANGE: "نشاط معتدل يصلح لرصد التراكم",
    MarketRegime.QUIET_LOW_LIQUIDITY: "هدوء منخفض السيولة",
}

_TREND_AR = {True: "صاعد", False: "هابط"}


def format_no_signal(reason: str) -> str:
    return f"لا توجد إشارة تراكم مؤهلة في هذا التشغيل. {reason}"


def _fmt_pct(x: float) -> str:
    return f"{x:+.1f}%"


def format_signal_report(
    exchange_name: str,
    market_timing: MarketTimingResult,
    verdict: SignalVerdict,
    trade_plan: Optional[TradePlan],
) -> str:
    cand = verdict.confirmed.candidate
    snap = market_timing.snapshot
    lines: List[str] = []

    lines.append("أولا، قراءة توقيت السوق.")
    lines.append(
        f"المتوسط المرجح بالحجم لتغير الأسعار خلال 24 ساعة على {exchange_name} "
        f"{_fmt_pct(snap.volume_weighted_change_pct_24h)}، "
        f"وحصة البيتكوين من حجم التداول نحو {snap.btc_quote_volume_share_pct:.1f} بالمئة، "
        f"والسوق في حالة {_REGIME_AR[market_timing.regime]}."
    )
    lines.append("")

    lines.append("ثانيا، العملة المختارة وسبب اختيارها.")
    lines.append(
        f"العملة {cand.base_asset} مقابل {cand.quote_asset}، "
        f"بأداء شهري قريب من التسطح {_fmt_pct(cand.perf_30d_pct)}، "
        f"وهذا يفرقها عن عملات تحركت بالفعل بنسب كبيرة خلال الشهر الماضي."
    )
    lines.append("")

    lines.append("ثالثا، معايير التراكم المتحققة.")
    c = verdict.compression
    met = []
    if c.adx_low:
        met.append(f"ADX منخفض عند {c.adx_value:.1f}")
    if c.bb_squeezed:
        met.append("نطاقات بولينجر ضيقة نسبيا مقارنة بتاريخها القريب")
    if c.rsi_neutral:
        met.append(f"RSI محايد عند {c.rsi_value:.1f}")
    if c.obv_flat_or_rising:
        met.append("حجم التداول التراكمي مستقر أو مرتفع رغم تسطح السعر")
    lines.append("، ".join(met) + ".")
    lines.append("")

    lines.append("رابعا، جدول المؤشرات عبر الأطر الزمنية.")
    for tf_name, read in verdict.confirmed.reads.items():
        last = read.last
        rsi_v = last["rsi14"]
        macd_h = last["macd_hist"]
        adx_v = last["adx"]
        bias = "صاعد" if macd_h == macd_h and macd_h > 0 else "هابط" if macd_h == macd_h else "محايد"
        lines.append(
            f"إطار {tf_name}، RSI {rsi_v:.1f}، ADX {adx_v:.1f}، "
            f"هيستوجرام MACD {bias}."
        )
    lines.append("")

    lines.append("خامسا، إشارة الدخول.")
    trig = verdict.trigger
    lines.append(
        f"الإشارة قائمة الآن بدرجة قناعة {verdict.conviction}، "
        f"بسبب {('، '.join(trig.reasons) if trig.reasons else 'تأكيد مباشر من الأطر الزمنية')}."
    )
    lines.append("")

    lines.append("سادسا، خطة الصفقة.")
    if trade_plan is not None:
        targets_str = "، ".join(
            f"{t:.6g} بعائد إلى مخاطرة {rr:.2f}"
            for t, rr in zip(trade_plan.targets, trade_plan.reward_risk)
        )
        basis_ar = "أقرب دعم فعلي" if trade_plan.stop_basis == "structure" else "مضاعف المدى الحقيقي المتوسط"
        lines.append(
            f"سعر الدخول {trade_plan.entry:.6g}، "
            f"وقف الخسارة {trade_plan.stop:.6g} بالاعتماد على {basis_ar}، "
            f"والأهداف {targets_str}."
        )
    else:
        lines.append("تعذر بناء خطة صفقة موثوقة لغياب بيانات كافية لحساب المدى الحقيقي المتوسط.")
    lines.append("")

    lines.append(
        "سابعا، هذه قراءة فنية مبنية على بيانات السوق العامة وليست نصيحة مالية ولا ضمانا لنتيجة."
    )

    return "\n".join(lines)
