# ComfyUI-AdvancedAlphaProcessor

## これは何？
これはComfyUIのカスタムノードです。
- 入力画像の白い部分を輝度で透明に抜く
- 黒と透明で出力する
- グレーと透明で出力する  
が可能のシンプルなノードです。  
連番処理可能。

![image](https://github.com/852wa/ComfyUI-AAP/blob/master/sample.png)


- invert_alpha  
  反転する（デフォルトはオン）Output出力から出る
- midrange_cut  
  グレーカットする（デフォルトはオフ）Output出力から結果が出る。
- cut_threshold  
  midrange_cutのしきい値調整（midrange_cutがオフの場合は無効）
- gamma_correction  
  アルファチャンネル生成前の画像の明るさの調整。値を大きくすると明るい部分がより明るく暗い部分がより暗くなり、値を小さくすると中間調のコントラストが強調される。
- remove_black_threshold  
  Output出力の黒部分をカットしてグレーを残すしきい値調整。Black Removed出力から出る。（midrange_cutは基本オフで使用する）
- force_grayscale  
  グレースケールで実行する（デフォルトはオン　オンのほうが精度がいい）
