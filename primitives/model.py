import tensorflow as tf

class EncoderDecoder(tf.keras.models.Model):
    
    def __init__(self, L, m, c, d):
        super().__init__()

        self.encoder = tf.keras.layers.LSTM(units=c, dropout=d, return_state=True)
        # デコーダはエンコーダの最終状態を受け取る等の制御が必要になるため，cellで定義する
        self.decoder = tf.keras.layers.LSTMCell(units=c, dropout=d)
        self.outputs = tf.keras.layers.Dense(units=m)

        self.L = L
        self.m = m
        self.c = c
    
    def call(self, inputs, training=True):
        # ※　inputs.shape=(None, 100, 2)
        # デコーダのドロップアウトマスクとリカレントドロップアウトマスクをリセット
        # 各バッチまたはエポックでドロップアウトをランダムに適用するために使用
        self.decoder.reset_dropout_mask()
        self.decoder.reset_recurrent_dropout_mask()
        
        # エンコーダに入力データ（inputs）を供給し、隠れ状態（he）とセル状態（ce）を取得
        _, he, ce = self.encoder(inputs)
        # ※ he.shape=(None, 64)
        # →バッチ内のシーケンスごとに隠れ状態とセル状態がある

        # エンコーダの最終隠れ状態とセル状態をデコーダの初期状態に設定するために用意
        hd = tf.identity(he) # 入力と同じ形状と内容を持つ Tensor を返す
        cd = tf.identity(ce)

        # デコーダが各タイムステップで生成した出力を格納するための変数
        # それぞれのindexがタイムステップに該当，
        # valueがbatch内のそれぞれのシーケンス時の該当タイムステップの値
        r = tf.TensorArray(
            element_shape=(inputs.shape[0], self.m),
            size=self.L,
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )
    
        # 論文参照
        # エンコーダーとデコーダーは、逆順に時系列を再構成するように学習されるため，
        # エンコーダから受け取った状態による出力は，最後のindexに格納する
        r = r.write(
            index=self.L - 1,
            value=self.outputs(hd)
        )
       # ※　r.shape=(100, None, 2)

        # エンコーダーとデコーダーは、逆順に時系列を再構成するように学習される
        for t in tf.range(start=self.L - 2, limit=-1, delta=-1):
            # 訓練時はinputのタイムステップを１つ取り出して入力
            # 推論時は，一つ前の推論結果を入力する
            # ※shape=(None, 2)
            _, [hd, cd] = self.decoder(
                inputs=inputs[:, t + 1, :] if training else r.read(index=t + 1),
                states=[hd, cd],
                training=training
            )
        
            r = r.write(
                index=t,
                value=self.outputs(hd)
            )
        # 形状を（バッチ，タイムステップ，特徴量）に整形
        return tf.transpose(r.stack(), (1, 0, 2))