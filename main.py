import os
import io
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import mplfinance as mpf
from datetime import datetime, timezone
import anthropic

# ─────────────────────────────────────────
# 환경변수
# ─────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")   # 채널 ID (예: -100xxxxxxxxxx)
SYMBOL             = os.environ.get("SYMBOL", "BTCUSDT")
MIN_SIGNAL_SCORE   = float(os.environ.get("MIN_SIGNAL_SCORE", "6"))  # 6점 이상만 발송


# ─────────────────────────────────────────
# 1. 바이낸스 캔들 데이터 수집
# ─────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    # Bybit API (지역 제한 없음)
    interval_map = {"1h": "60", "4h": "240", "1d": "D"}
    bybit_interval = interval_map.get(interval, "60")
    
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["result"]["list"]
    
    # Bybit는 최신순으로 내려줌 → 역순 정렬
    data = list(reversed(data))
    
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    return df


# ─────────────────────────────────────────
# 2. 보조지표 계산
# ─────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # 볼린저밴드
    sma20         = df["close"].rolling(20).mean()
    std20         = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    # ATR
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    return df


def summarize_indicators(df1h: pd.DataFrame, df4h: pd.DataFrame) -> dict:
    """최신 캔들 기준 지표 요약"""
    def last(df):
        r = df.iloc[-1]
        prev = df.iloc[-2]
        return {
            "close":       round(r["close"], 2),
            "ema20":       round(r["ema20"], 2),
            "ema50":       round(r["ema50"], 2),
            "ema200":      round(r["ema200"], 2),
            "rsi":         round(r["rsi"], 1),
            "macd":        round(r["macd"], 4),
            "macd_signal": round(r["macd_signal"], 4),
            "macd_hist":   round(r["macd_hist"], 4),
            "bb_upper":    round(r["bb_upper"], 2),
            "bb_lower":    round(r["bb_lower"], 2),
            "atr":         round(r["atr"], 2),
            "volume":      round(r["volume"], 2),
            "vol_avg20":   round(df["volume"].tail(20).mean(), 2),
            "ema_order":   "정배열" if r["ema20"] > r["ema50"] > r["ema200"] else
                           "역배열" if r["ema20"] < r["ema50"] < r["ema200"] else "혼조",
            "ema20_cross": "골든크로스" if r["ema20"] > r["ema50"] and prev["ema20"] <= prev["ema50"]
                           else "데드크로스" if r["ema20"] < r["ema50"] and prev["ema20"] >= prev["ema50"]
                           else "유지",
        }
    return {"1h": last(df1h), "4h": last(df4h)}


# ─────────────────────────────────────────
# 3. Claude API로 신호 분석
# ─────────────────────────────────────────
def analyze_with_claude(summary: dict, symbol: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system_prompt = """당신은 코인 선물 트레이딩 전문 AI 애널리스트입니다.
주어진 멀티타임프레임 지표 데이터를 기반으로 신호를 분석하고,
반드시 아래 JSON 형식으로만 응답하세요. JSON 외 다른 텍스트는 절대 포함하지 마세요.

{
  "direction": "LONG" | "SHORT" | "WAIT",
  "score": 1~10 (신호 강도, 10이 최강),
  "entry": 진입가격(숫자),
  "tp1": 1차 목표가(숫자),
  "tp2": 2차 목표가(숫자),
  "sl": 손절가(숫자),
  "leverage": 권장 레버리지(숫자, 3~20),
  "trend_4h": "강한상승" | "상승" | "횡보" | "하락" | "강한하락",
  "trend_1h": "강한상승" | "상승" | "횡보" | "하락" | "강한하락",
  "reasons": ["근거1", "근거2", "근거3", "근거4"],
  "risk": "낮음" | "중간" | "높음",
  "summary": "한줄 시장 코멘트 (50자 이내)"
}

score 산정 기준:
- 8~10: 명확한 추세 + 복수 지표 일치 → 강한 신호
- 6~7: 추세 있음 + 일부 지표 일치 → 발송 가능
- 1~5: 불확실 / 횡보 → WAIT 권장"""

    user_msg = f"""심볼: {symbol}
분석 시각(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}

=== 4시간봉 지표 ===
현재가: {summary['4h']['close']}
EMA20: {summary['4h']['ema20']} | EMA50: {summary['4h']['ema50']} | EMA200: {summary['4h']['ema200']}
EMA 정렬: {summary['4h']['ema_order']} | EMA 크로스: {summary['4h']['ema20_cross']}
RSI(14): {summary['4h']['rsi']}
MACD: {summary['4h']['macd']} | Signal: {summary['4h']['macd_signal']} | Hist: {summary['4h']['macd_hist']}
볼린저밴드 상단: {summary['4h']['bb_upper']} | 하단: {summary['4h']['bb_lower']}
ATR(14): {summary['4h']['atr']}
거래량: {summary['4h']['volume']} (20봉 평균: {summary['4h']['vol_avg20']})

=== 1시간봉 지표 ===
현재가: {summary['1h']['close']}
EMA20: {summary['1h']['ema20']} | EMA50: {summary['1h']['ema50']}
EMA 정렬: {summary['1h']['ema_order']} | EMA 크로스: {summary['1h']['ema20_cross']}
RSI(14): {summary['1h']['rsi']}
MACD: {summary['1h']['macd']} | Signal: {summary['1h']['macd_signal']} | Hist: {summary['1h']['macd_hist']}
볼린저밴드 상단: {summary['1h']['bb_upper']} | 하단: {summary['1h']['bb_lower']}
ATR(14): {summary['1h']['atr']}
거래량: {summary['1h']['volume']} (20봉 평균: {summary['1h']['vol_avg20']})"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = message.content[0].text.strip()
    # JSON 파싱
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────
# 4. 차트 이미지 생성
# ─────────────────────────────────────────
def generate_chart(df: pd.DataFrame, signal: dict, symbol: str) -> bytes:
    df_chart = df.tail(60).copy()

    # mplfinance 스타일
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit",
        wick={"up": "#26a69a", "down": "#ef5350"},
        volume={"up": "#26a69a55", "down": "#ef535055"}
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor="#131722",
        edgecolor="#2a2e39",
        figcolor="#131722",
        gridcolor="#2a2e39",
        gridstyle="--",
        rc={"axes.labelcolor": "#d1d4dc", "xtick.color": "#787b86", "ytick.color": "#787b86"}
    )

    # 추가선
    add_plots = [
        mpf.make_addplot(df_chart["ema20"],  color="#f7c948", width=1.2, label="EMA20"),
        mpf.make_addplot(df_chart["ema50"],  color="#2196f3", width=1.2, label="EMA50"),
        mpf.make_addplot(df_chart["ema200"], color="#ff6d00", width=1.2, label="EMA200"),
        mpf.make_addplot(df_chart["bb_upper"], color="#9c27b088", width=0.8, linestyle="--"),
        mpf.make_addplot(df_chart["bb_lower"], color="#9c27b088", width=0.8, linestyle="--"),
    ]

    fig, axes = mpf.plot(
        df_chart,
        type="candle",
        style=style,
        addplot=add_plots,
        volume=True,
        figsize=(12, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=200
    )

    ax = axes[0]

    # TP/SL 수평선
    direction = signal.get("direction", "WAIT")
    entry = signal.get("entry")
    tp1   = signal.get("tp1")
    tp2   = signal.get("tp2")
    sl    = signal.get("sl")

    if direction != "WAIT" and entry:
        ax.axhline(entry, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.axhline(tp1,   color="#26a69a", linewidth=1.2, linestyle="-",  alpha=0.9)
        ax.axhline(tp2,   color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(sl,    color="#ef5350", linewidth=1.2, linestyle="-",  alpha=0.9)

        xmax = len(df_chart) - 1
        ax.text(xmax, tp1,   f" TP1 {tp1:,.0f}", color="#26a69a", fontsize=8, va="center")
        ax.text(xmax, tp2,   f" TP2 {tp2:,.0f}", color="#26a69a", fontsize=8, va="center")
        ax.text(xmax, sl,    f" SL {sl:,.0f}",   color="#ef5350", fontsize=8, va="center")
        ax.text(xmax, entry, f" 진입 {entry:,.0f}", color="#ffffff", fontsize=8, va="center")

    # 제목
    color = "#26a69a" if direction == "LONG" else "#ef5350" if direction == "SHORT" else "#787b86"
    fig.suptitle(
        f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT' if direction=='SHORT' else '⏸ WAIT'}  |  {symbol}  |  4H",
        color=color, fontsize=14, fontweight="bold", y=0.98
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────
# 5. 텔레그램 발송
# ─────────────────────────────────────────
def format_message(signal: dict, symbol: str) -> str:
    direction = signal.get("direction", "WAIT")
    score     = signal.get("score", 0)
    entry     = signal.get("entry", 0)
    tp1       = signal.get("tp1", 0)
    tp2       = signal.get("tp2", 0)
    sl        = signal.get("sl", 0)
    leverage  = signal.get("leverage", 5)
    trend_4h  = signal.get("trend_4h", "-")
    trend_1h  = signal.get("trend_1h", "-")
    reasons   = signal.get("reasons", [])
    risk      = signal.get("risk", "-")
    summary   = signal.get("summary", "")

    if direction == "LONG":
        dir_emoji = "🟢"
        dir_label = "LONG (매수)"
        rr = round(abs(tp1 - entry) / abs(entry - sl), 2) if entry != sl else 0
    elif direction == "SHORT":
        dir_emoji = "🔴"
        dir_label = "SHORT (매도)"
        rr = round(abs(entry - tp1) / abs(sl - entry), 2) if entry != sl else 0
    else:
        dir_emoji = "⏸"
        dir_label = "WAIT (관망)"
        rr = 0

    stars = "⭐" * min(score, 10)
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    msg = f"""{dir_emoji} *{dir_label}* | `{symbol}` | Futures

━━━━━━━━━━━━━━━
📊 *신호 강도* {stars} ({score}/10)
🕐 *4H 추세:* {trend_4h}  |  *1H 추세:* {trend_1h}

━━━━━━━━━━━━━━━
📍 *진입가:* `${entry:,.2f}`
🎯 *TP1:* `${tp1:,.2f}`
🎯 *TP2:* `${tp2:,.2f}`
🛡 *SL:* `${sl:,.2f}`
⚡ *권장 레버리지:* {leverage}x
📐 *R:R =* 1 : {rr}

━━━━━━━━━━━━━━━
📝 *분석 근거*
"""
    for i, reason in enumerate(reasons, 1):
        msg += f"  {i}. {reason}\n"

    msg += f"""
━━━━━━━━━━━━━━━
💬 _{summary}_

⚠️ 리스크: *{risk}*
🕐 _{now}_

#FLOW #신호 #{symbol}"""
    return msg


def send_telegram(text: str, photo: bytes, bot_token: str, chat_id: str):
    base = f"https://api.telegram.org/bot{bot_token}"

    # 사진 + 캡션 발송
    files = {"photo": ("chart.png", photo, "image/png")}
    data  = {
        "chat_id":    chat_id,
        "caption":    text,
        "parse_mode": "Markdown"
    }
    r = requests.post(f"{base}/sendPhoto", data=data, files=files, timeout=30)
    r.raise_for_status()
    print(f"[텔레그램] 전송 완료: {r.json().get('ok')}")


# ─────────────────────────────────────────
# 6. 메인 실행
# ─────────────────────────────────────────
def main():
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] 신호 분석 시작 → {SYMBOL}")

    # 데이터 수집
    print("  📡 바이낸스 데이터 수집 중...")
    df1h = add_indicators(fetch_candles(SYMBOL, "1h", limit=220))
    df4h = add_indicators(fetch_candles(SYMBOL, "4h", limit=220))

    # 지표 요약
    summary = summarize_indicators(df1h, df4h)
    print(f"  💰 현재가: ${summary['4h']['close']:,} | RSI(4H): {summary['4h']['rsi']}")

    # Claude 분석
    print("  🤖 Claude 분석 중...")
    signal = analyze_with_claude(summary, SYMBOL)
    direction = signal.get("direction", "WAIT")
    score     = signal.get("score", 0)
    print(f"  📊 결과: {direction} | 점수: {score}/10")

    # 신호 강도 필터
    if score < MIN_SIGNAL_SCORE:
        print(f"  ⏸ 신호 미달 (direction={direction}, score={score}) → 발송 건너뜀")
        return

    # 차트 생성
    print("  🖼 차트 생성 중...")
    chart_img = generate_chart(df4h, signal, SYMBOL)

    # 메시지 포맷
    message = format_message(signal, SYMBOL)

    # 텔레그램 발송
    print("  📨 텔레그램 발송 중...")
    send_telegram(message, chart_img, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    print("  ✅ 완료!")


if __name__ == "__main__":
    main()
