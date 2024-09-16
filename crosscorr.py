from keras.layers import Layer
import keras.backend as K


class CrossCorrelation(Layer):
    def __init__(self,
                 scale=1,
                 normalization=True,
                 epsilon=1e-5,
                 **kwargs):
        super(CrossCorrelation, self).__init__(**kwargs)
        self.normalization = normalization
        self.scale = scale
        self.epsilon = epsilon

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][2])
        return output_shape

    # def build(self, input_shape):
    #     super(CrossCorrelation, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        k = inputs[0]
        q = inputs[1]
        v = inputs[2]

        if self.normalization:
            mean = K.mean(k, axis=1, keepdims=True)
            std = K.std(k, axis=1, keepdims=True)
            k = (k - mean) / (std + self.epsilon)

            mean = K.mean(q, axis=1, keepdims=True)
            std = K.std(q, axis=1, keepdims=True)
            q = (q - mean) / (std + self.epsilon)

        q = q[:, -1]
        wt = K.batch_dot(k, q) / K.int_shape(k)[1]
        wt *= self.scale
        wt = K.exp(wt)
        v = v[:, 1:]
        wt = wt[:, :-1]
        wt /= K.sum(wt, axis=1, keepdims=True)
        weighted_input = v * K.expand_dims(wt)
        output = K.sum(weighted_input, axis=1)
        return output
