# SEAP2vecの設計

## ToDo
- ADDAの内容確認, modelクラスを伏せて実装  
- BTとVitの連携  

## 方針
- パーツごとに作成する  
- ADDA-BT-ViTの形にする  
- Vit > BT > ADDAの順に作成する  
- ADDA: domain adaptation  
- BT: Barlow Twins  
- ViT: backbone for BT  
- BTの中にViTを入れる  
- ADDAの中にBTを入れる  
- ADDAがmainの階層になる  
- ADDAにてdata作成を行う  

## training方法
- ViT  
    - labelありで組む  
    - クラス数を使っているところがないか確認  
    - BTの中で組む  
- BT  
    - label使わない  
    - ADDAの中で組む  
- ADDA  
    - 要確認  

## data作成
いくらかアプローチがある  

### 決定事項
1. 1つのSEAPサンプルから2度sampleしてBTに供する, Transform使う  
2. 1つのSEAPサンプルから2度sampleしてBTに供する, Transform使わない  
3. 1つのSEAPサンプルから1度sampleし, ヒストグラムの粒度を変えた後, BTに供する, Transform使う  
4. 1つのSEAPサンプルから1度sampleし, ヒストグラムの粒度を変えた後, BTに供する, Transform使わない  
5. 1つのSEAPサンプルから2度sampleし, ヒストグラムの粒度を変えた後, BTに供する, Transform使う  
6. 1つのSEAPサンプルから2度sampleし, ヒストグラムの粒度を変えた後, BTに供する, Transform使わない  

- List item 1
- List item 2
- List item 3

**Bold text**

*Italic text*

[Link to Google](https://www.google.com)

![Image](image.png)