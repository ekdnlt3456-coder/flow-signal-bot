import os
import io
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timezone
import anthropic

# ─────────────────────────────────────────
# 환경변수
# ─────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOL             = os.environ.get("SYMBOL", "BTCUSDT")
MIN_SIGNAL_SCORE   = float(os.environ.get("MIN_SIGNAL_SCORE", "6"))


# ─────────────────────────────────────────
# 1. 바이비트 캔들 데이터 수집
# ─────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    interval_map = {"1h": "60", "4h": "240", "1d": "D"}
    bybit_interval = interval_map.get(interval, "60")
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = list(reversed(r.json()["result"]["list"]))
    df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","turnover"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    return df


# ─────────────────────────────────────────
# 2. 보조지표 계산
# ─────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    sma20          = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20

    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    return df


def summarize_indicators(df1h: pd.DataFrame, df4h: pd.DataFrame) -> dict:
    def last(df):
        r    = df.iloc[-1]
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
# 3. 실시간 뉴스/이슈 수집 (웹서치)
# ─────────────────────────────────────────
def fetch_market_news(client: anthropic.Anthropic, symbol: str) -> str:
    print("  🌐 실시간 뉴스/이슈 검색 중...")
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"""지금 당장 {symbol} 비트코인 시장에 영향을 줄 수 있는 실시간 뉴스와 이슈를 검색해줘.
다음 항목들을 중심으로:
1. FOMC, CPI, PPI 등 주요 경제지표 발표 일정
2. 비트코인/크립토 관련 주요 뉴스
3. 고래 움직임, 대규모 청산, 거래소 이슈
4. 미국 증시 및 달러 인덱스 동향

결과를 아래 형식으로만 답해줘 (JSON):
{{
  "news_items": ["이슈1 (출처)", "이슈2 (출처)", "이슈3 (출처)"],
  "caution": ["주의사항1", "주의사항2"],
  "market_sentiment": "공포" | "중립" | "탐욕"
}}
JSON만 출력, 다른 텍스트 없이."""
            }]
        )

        # 응답에서 텍스트 추출
        full_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                full_text += block.text

        clean = full_text.replace("```json", "").replace("```", "").strip()
        news_data = json.loads(clean)
        return news_data

    except Exception as e:
        print(f"  ⚠️ 뉴스 검색 실패 (기본값 사용): {e}")
        return {
            "news_items": ["실시간 뉴스 수집 불가 - 지표 기반 분석만 적용"],
            "caution": ["뉴스 확인 후 진입 권장"],
            "market_sentiment": "중립"
        }


# ─────────────────────────────────────────
# 4. Claude API로 신호 분석
# ─────────────────────────────────────────
def analyze_with_claude(summary: dict, symbol: str, news_data: dict) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 뉴스 먼저 수집
    if not news_data:
        news_data = fetch_market_news(client, symbol)

    news_text = "\n".join([f"• {n}" for n in news_data.get("news_items", [])])
    caution_text = "\n".join([f"• {c}" for c in news_data.get("caution", [])])
    sentiment = news_data.get("market_sentiment", "중립")

    system_prompt = """당신은 코인 선물 트레이딩 전문 AI 애널리스트입니다.
기술적 지표 + 실시간 뉴스/이슈를 종합해서 신호를 분석합니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외 다른 텍스트는 절대 포함하지 마세요.

{
  "direction": "LONG" | "SHORT" | "WAIT",
  "score": 1~10 (신호 강도),
  "entry": 진입가격(숫자),
  "tp1": 1차 목표가(숫자),
  "tp2": 2차 목표가(숫자),
  "sl": 손절가(숫자),
  "leverage": 권장 레버리지(숫자, 3~20),
  "trend_4h": "강한상승" | "상승" | "횡보" | "하락" | "강한하락",
  "trend_1h": "강한상승" | "상승" | "횡보" | "하락" | "강한하락",
  "reasons": ["쉬운말 근거1", "쉬운말 근거2", "쉬운말 근거3", "쉬운말 근거4"],
  "issues": ["주의사항1", "주의사항2", "주의사항3"],
  "risk": "낮음" | "중간" | "높음",
  "summary": "한줄 시장 코멘트 (50자 이내)"
}

중요 규칙:
- reasons는 차트 전문용어 금지. 초보자도 이해할 수 있는 쉬운 말로.
  예) "EMA 골든크로스" → "단기 평균선이 중기선 위로 올라서는 중 (상승 신호)"
  예) "RSI 과매수" → "현재 너무 많이 오른 상태, 조정 가능성 있음"
  예) "볼밴 상단 터치" → "가격이 상승 한계선에 닿은 상태"
- issues는 뉴스 이슈 + 차트 리스크 포함
- score 기준: 8~10 강한신호, 6~7 발송가능, 1~5 WAIT"""

    user_msg = f"""심볼: {symbol}
분석 시각(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}
시장 심리: {sentiment}

=== 실시간 뉴스/이슈 ===
{news_text}

=== 주의사항 ===
{caution_text}

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
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────
# 5. 차트 이미지 생성
# ─────────────────────────────────────────
def generate_chart(df: pd.DataFrame, signal: dict, symbol: str) -> bytes:
    df_chart = df.tail(60).copy()

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

    add_plots = [
        mpf.make_addplot(df_chart["ema20"],   color="#f7c948", width=1.2, label="EMA20"),
        mpf.make_addplot(df_chart["ema50"],   color="#2196f3", width=1.2, label="EMA50"),
        mpf.make_addplot(df_chart["ema200"],  color="#ff6d00", width=1.2, label="EMA200"),
        mpf.make_addplot(df_chart["bb_upper"],color="#9c27b088", width=0.8, linestyle="--"),
        mpf.make_addplot(df_chart["bb_lower"],color="#9c27b088", width=0.8, linestyle="--"),
    ]

    fig, axes = mpf.plot(
        df_chart, type="candle", style=style, addplot=add_plots,
        volume=True, figsize=(12, 7), returnfig=True,
        tight_layout=True, warn_too_much_data=200
    )

    ax = axes[0]
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

        # TP/SL % 계산
        tp1_pct = round((tp1 - entry) / entry * 100, 1) if direction == "LONG" else round((entry - tp1) / entry * 100, 1)
        tp2_pct = round((tp2 - entry) / entry * 100, 1) if direction == "LONG" else round((entry - tp2) / entry * 100, 1)
        sl_pct  = round((entry - sl)  / entry * 100, 1) if direction == "LONG" else round((sl - entry)  / entry * 100, 1)

        ax.text(xmax, tp1,   f" TP1 {tp1:,.0f} (+{tp1_pct}%)", color="#26a69a", fontsize=8, va="center")
        ax.text(xmax, tp2,   f" TP2 {tp2:,.0f} (+{tp2_pct}%)", color="#26a69a", fontsize=8, va="center")
        ax.text(xmax, sl,    f" SL {sl:,.0f} (-{sl_pct}%)",    color="#ef5350", fontsize=8, va="center")
        ax.text(xmax, entry, f" 진입 {entry:,.0f}",             color="#ffffff", fontsize=8, va="center")

    color = "#26a69a" if direction == "LONG" else "#ef5350" if direction == "SHORT" else "#787b86"
    fig.suptitle(
        f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT' if direction=='SHORT' else '⏸ WAIT'}  |  {symbol}  |  4H  |  AI 분석",
        color=color, fontsize=14, fontweight="bold", y=0.98
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────
# 6. 메시지 포맷
# ─────────────────────────────────────────
def format_message(signal: dict, symbol: str, news_data: dict) -> str:
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
    issues    = signal.get("issues", [])
    risk      = signal.get("risk", "-")
    summary_t = signal.get("summary", "")
    sentiment = news_data.get("market_sentiment", "중립")

    sentiment_emoji = {"공포": "😨", "중립": "😐", "탐욕": "🤑"}.get(sentiment, "😐")

    if direction == "LONG":
        dir_emoji = "🟢"
        dir_label = "LONG (매수)"
        tp1_pct = round((tp1 - entry) / entry * 100, 1)
        tp2_pct = round((tp2 - entry) / entry * 100, 1)
        sl_pct  = round((entry - sl)  / entry * 100, 1)
        rr = round(abs(tp1 - entry) / abs(entry - sl), 2) if entry != sl else 0
    elif direction == "SHORT":
        dir_emoji = "🔴"
        dir_label = "SHORT (공매도)"
        tp1_pct = round((entry - tp1) / entry * 100, 1)
        tp2_pct = round((entry - tp2) / entry * 100, 1)
        sl_pct  = round((sl - entry)  / entry * 100, 1)
        rr = round(abs(entry - tp1) / abs(sl - entry), 2) if entry != sl else 0
    else:
        dir_emoji = "⏸"
        dir_label = "WAIT (관망)"
        tp1_pct = tp2_pct = sl_pct = rr = 0

    stars = "⭐" * min(score, 10)
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    msg = f"""{dir_emoji} *{dir_label}* | `{symbol}` | Futures

━━━━━━━━━━━━━━━
📊 *신호 강도* {stars} ({score}/10)
🕐 *4H 추세:* {trend_4h}  |  *1H 추세:* {trend_1h}
{sentiment_emoji} *시장 심리:* {sentiment}

━━━━━━━━━━━━━━━
📍 *진입가:* `${entry:,.2f}`
🎯 *TP1:* `${tp1:,.2f}` (+{tp1_pct}%)
🎯 *TP2:* `${tp2:,.2f}` (+{tp2_pct}%)
🛡 *손절가:* `${sl:,.2f}` (-{sl_pct}%)
⚡ *권장 레버리지:* {leverage}x
📐 *R:R =* 1 : {rr}

━━━━━━━━━━━━━━━
📝 *분석 근거*
"""
    for i, reason in enumerate(reasons, 1):
        msg += f"  {i}. {reason}\n"

    msg += "\n━━━━━━━━━━━━━━━\n⚠️ *주의사항 / 이슈*\n"
    for issue in issues:
        msg += f"  • {issue}\n"

    msg += f"""
━━━━━━━━━━━━━━━
💬 _{summary_t}_

⚠️ 리스크: *{risk}*
🕐 _{now}_

━━━━━━━━━━━━━━━
_본 신호는 AI가 생성한 참고용 정보입니다._
_투자 결과에 대한 책임은 본인에게 있으며,_
_투자 원금 손실이 발생할 수 있습니다._

#FLOW #신호 #{symbol}"""
    return msg


# ─────────────────────────────────────────
# 7. 텔레그램 발송
# ─────────────────────────────────────────
def send_telegram(text: str, photo: bytes, bot_token: str, chat_id: str):
    base  = f"https://api.telegram.org/bot{bot_token}"
    files = {"photo": ("chart.png", photo, "image/png")}
    data  = {"chat_id": chat_id, "caption": text, "parse_mode": "Markdown"}
    r = requests.post(f"{base}/sendPhoto", data=data, files=files, timeout=30)
    r.raise_for_status()
    print(f"  [텔레그램] 전송 완료: {r.json().get('ok')}")


# ─────────────────────────────────────────
# 8. 메인 실행
# ─────────────────────────────────────────
def main():
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] 신호 분석 시작 → {SYMBOL}")

    print("  📡 데이터 수집 중...")
    df1h = add_indicators(fetch_candles(SYMBOL, "1h", limit=220))
    df4h = add_indicators(fetch_candles(SYMBOL, "4h", limit=220))
    summary = summarize_indicators(df1h, df4h)
    print(f"  💰 현재가: ${summary['4h']['close']:,} | RSI(4H): {summary['4h']['rsi']}")

    # 실시간 뉴스 수집
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    news_data = fetch_market_news(client, SYMBOL)
    print(f"  📰 시장 심리: {news_data.get('market_sentiment', '-')}")

    # Claude 분석
    print("  🤖 Claude 분석 중...")
    signal    = analyze_with_claude(summary, SYMBOL, news_data)
    direction = signal.get("direction", "WAIT")
    score     = signal.get("score", 0)
    print(f"  📊 결과: {direction} | 점수: {score}/10")

    # 신호 강도 필터
    if direction == "WAIT" or score < MIN_SIGNAL_SCORE:
        print(f"  ⏸ 신호 미달 (direction={direction}, score={score}) → 발송 건너뜀")
        return

    # 차트 생성
    print("  🖼 차트 생성 중...")
    chart_img = generate_chart(df4h, signal, SYMBOL)

    # 메시지 포맷 + 발송
    message = format_message(signal, SYMBOL, news_data)
    print("  📨 텔레그램 발송 중...")
    send_telegram(message, chart_img, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    print("  ✅ 완료!")


if __name__ == "__main__":
    main()
