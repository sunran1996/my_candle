# YH02 红利低波 BB+RSI 策略

## 参数
- BB(45, 2.0) + RSI(14, 30, 70)
- 扩张卖出: BB上轨 ∩ RSI≥65 + BB加速度过滤 + 价格覆写
- 收缩买卖: BB ∪ RSI
- 无DCA纯策略收益: +378.8%  (2019-2026)
- Walk-Forward OOS最差Sharpe: 1.47

## 文件
- `monitor.py` — 主脚本 (默认实时信号, --from 回测)
- `_walkforward.py` — Walk-Forward验证
- `compare_final.png` — YH01 vs YH02对比图
