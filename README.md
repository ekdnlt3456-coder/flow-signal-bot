# 🤖 FLOW 코인 선물 신호 봇

바이낸스 선물 데이터 → Claude AI 분석 → 텔레그램 자동 발송

## 파일 구조
```
crypto-signal-bot/
├── main.py           # 메인 실행 파일
├── requirements.txt  # 라이브러리 목록
├── railway.toml      # Railway 설정
└── cron.json         # 크론 스케줄 설정
```

## Railway 환경변수 설정 (필수)

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API 키 | sk-ant-... |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | 123456:ABC... |
| `TELEGRAM_CHAT_ID` | 채널/그룹 ID | -100123456789 |
| `SYMBOL` | 분석 심볼 (선택) | BTCUSDT |
| `MIN_SIGNAL_SCORE` | 최소 발송 점수 (선택) | 6 |

## Railway 배포 순서

1. GitHub에 이 폴더 업로드 (새 레포 생성)
2. Railway → New Project → Deploy from GitHub repo
3. Variables 탭에서 위 환경변수 입력
4. Settings → Cron Schedule: `0 * * * *` (매 정시 실행)

## 신호 발송 조건
- direction이 LONG 또는 SHORT일 때
- score가 MIN_SIGNAL_SCORE (기본 6점) 이상일 때
- 두 조건 모두 충족해야 텔레그램으로 발송됨

## 텔레그램 채널 ID 확인 방법
1. 봇을 채널 관리자로 추가
2. 채널에 메시지 전송
3. `https://api.telegram.org/bot{봇토큰}/getUpdates` 접속
4. `chat.id` 값 복사 (음수로 시작하는 숫자)
