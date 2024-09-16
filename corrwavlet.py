from crosscorr import CrossCorrelation
from keras.layers import Input, Conv1D, Lambda, Dense, LSTM, Add, concatenate, BatchNormalization
from keras.models import Model
from keras.activations import relu
import keras.backend as K


leakyrelu = lambda x: relu(x, alpha=0.2)


def direction_mse(y_current, alpha=1):
    def dummy_loss(y_true, y_pred):
        direction_check = K.cast(K.not_equal(K.sign(y_pred - y_current), K.sign(y_true - y_current)), K.floatx())
        direction_loss = K.mean((direction_check + alpha) * K.square((y_true - y_pred)), axis=-1)
        # direction_loss = K.print_tensor(direction_loss, 'direction_loss')
        return direction_loss
    return dummy_loss


class CorrelateWavelet:
    def build(self,
              wave_size,
              wave_scales,
              wave_filters,
              wave_kernel_size,
              using_timediff=False,
              conv_activation=None,
              using_autocross_feat=True,
              autoreg_feats=50,
              autoreg_activation=leakyrelu,
              using_cross_feat=True,
              crosscorr_scale=1,
              cross_feats=25,
              cross_activation=leakyrelu,
              direction_units=200,
              direction_activation='tanh'
              ):
        self.nb_level = wave_size[0]

        def extend2list(scalar):
            if not isinstance(scalar, list):
                scalar = [scalar]
            if len(scalar) == 1:
                scalar *= self.nb_level
            assert(len(scalar) == self.nb_level)
            return scalar

        wave_scales = extend2list(wave_scales)
        wave_kernel_size = extend2list(wave_kernel_size)
        autoreg_feats = extend2list(autoreg_feats)
        cross_feats = extend2list(cross_feats)

        wave_input = Input(wave_size, name='wave_input')
        co_wave_input = Input(wave_size, name='co_wave_input')

        # slice_level = Lambda(lambda x, level: x[:, level], name='slice_level')
        # slice_end = Lambda(lambda x: x[:, -1], name='slice_end')
        time_difference = Lambda(lambda x: x[:, 1:] - x[:, :-1], name='time_difference')
        slice_scale = Lambda(lambda x, level, f_len: x[:, level, -f_len:], name='slice_scale')
        expand_dim = Lambda(lambda x: K.expand_dims(x))

        autoreg_feature_list = []
        conv_feature_list = []
        co_conv_feature_list = []

        for i in range(self.nb_level):
            # slice_level.arguments = {'level': i}
            slice_scale.arguments = {'level': i, 'f_len': wave_kernel_size[i]}
            autoreg_feature = slice_scale(wave_input)
            autoreg_feature = Dense(autoreg_feats[i], activation=autoreg_activation)(autoreg_feature)
            # autoreg_feature = LSTM(autoreg_feats[i], activation=autoreg_activation)(expand_dim(autoreg_feature))
            autoreg_feature_list.append(autoreg_feature)

            slice_scale.arguments = {'level': i, 'f_len': wave_scales[i]}
            wave_feature = slice_scale(wave_input)
            if using_timediff:
                wave_feature = time_difference(wave_feature)
            wave_feature = expand_dim(wave_feature)
            conv_feature = Conv1D(filters=wave_filters, kernel_size=wave_kernel_size[i],
                                  activation=conv_activation)(wave_feature)
            conv_feature_list.append(conv_feature)

            wave_feature = slice_scale(co_wave_input)
            if using_timediff:
                wave_feature = time_difference(wave_feature)
            wave_feature = expand_dim(wave_feature)
            conv_feature = Conv1D(filters=wave_filters, kernel_size=wave_kernel_size[i],
                                  activation=conv_activation)(wave_feature)
            co_conv_feature_list.append(conv_feature)

        next_val_list = []
        cross_feat_list = []
        auto_crosscorrelation = CrossCorrelation(scale=crosscorr_scale)
        crosscorrelation = CrossCorrelation(scale=crosscorr_scale)
        for i in range(self.nb_level):
            layer_features = []
            if using_autocross_feat:
                autocorr_feat = auto_crosscorrelation(
                    [conv_feature_list[i], conv_feature_list[i], conv_feature_list[i]])
                layer_features.append(autocorr_feat)
            if using_cross_feat:
                for j in range(i, self.nb_level):
                    corr_feat = crosscorrelation(
                        [co_conv_feature_list[j], conv_feature_list[i], co_conv_feature_list[j]])
                    layer_features.append(corr_feat)
            # Predictor
            val_feature = autoreg_feature_list[i]
            if layer_features:
                concat_feature = concatenate(layer_features) if len(layer_features) > 1 else layer_features[0]
                final_cross_feats = Dense(cross_feats[i], activation=cross_activation)(concat_feature)
                cross_feat_list += layer_features
                # cross_feat_list.append(final_cross_feats)
                val_feature = concatenate([val_feature, final_cross_feats]);

            next_val = Dense(1, name='next_value_at_level_%d'%i)(val_feature)
            next_val_list.append(next_val)

        sum_val = Add(name='sum_value')(next_val_list) if len(next_val_list) > 1 else next_val_list[0]
        feature_list = autoreg_feature_list + cross_feat_list
        all_features = concatenate(feature_list) if len(feature_list) > 1 else feature_list[0]
        all_features = Dense(direction_units, activation=direction_activation)(all_features)
        updown = Dense(1, name='updown_value', activation='sigmoid')(all_features)

        next_val_list = [updown, sum_val] + next_val_list
        self.value_model = Model(inputs=[wave_input, co_wave_input], outputs=next_val_list, name='val_model')

    def compile(self, optimizer='adam', **kwargs):
        wave_input = self.value_model.inputs[0]
        y_current = K.sum(wave_input[:, :, -1], axis=-1, keepdims=True)
        losses = ['binary_crossentropy', direction_mse(y_current=y_current)]
        for i in range(self.nb_level):
            y_current = wave_input[:, i, -1]
            losses.append(direction_mse(y_current=y_current))

        self.value_model.compile(optimizer=optimizer, loss=losses, **kwargs)

