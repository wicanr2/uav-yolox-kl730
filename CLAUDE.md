# 目標
使用紅外線熱像儀
規劃在無人機上面直接辨識無人機, 使用 edge ai chip 完成這項工作
無人機 辨識 車輛/人

# 使用的晶片

- 預期KL730 (Kneron edge AI chip)
https://www.kneron.com/tw/news/blog/177/
耐能科技

# 使用算法
You Only Look Once (YOLO). (版本能動越新越好)

# 訓練資料集

- FLIR-ADAS

- FLIR Thermal Image 
https://oem.flir.com/en-hk/solutions/automotive/adas-dataset-form/

# ThermalUAV2UAV_Dataset

https://github.com/GabryV00/ThermalUAV2UAV_Dataset

# 該怎麼做

YOLO算法解釋
晶片選擇考量
軟體架構規劃
硬體整合建議 (基於KL730)
成本估算
重量
資料集要如何 train & test 
如何佈署到無人機上

允許分 agents 同步執行
先產生PLAN
