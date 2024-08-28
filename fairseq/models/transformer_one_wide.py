# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_iwslt_de_en,
)


@register_model("transformer_shared_ffn")
class TransformerSharedFfnModel(TransformerModel):

    #TODO: add different shared ffn options, EncNoDec, SharedEncDec etc.,
    #figure out how to add to config
    
    

    @classmethod
    def build_model(cls, args, task):
        
        transformer_model = TransformerModel.build_model(args, task)
        encoder = transformer_model.encoder
        decoder = transformer_model.decoder
        # Share encoder ffn 
        for encoder_layer_index in range(args.encoder_layers):
            base_encoder_layer = encoder.layers[0]
            if encoder_layer_index != 0:
                encoder_layer = encoder.layers[encoder_layer_index]
                encoder.layers[encoder_layer_index] = cls.copy_ffn(base_encoder_layer, encoder_layer)


        # Share decoder ffn 
        for decoder_layer_index in range(args.decoder_layers):
            base_decoder_layer = decoder.layers[0]
            if decoder_layer_index != 0:
                decoder_layer = decoder.layers[decoder_layer_index]
                decoder.layers[decoder_layer_index] = cls.copy_ffn(base_decoder_layer, decoder_layer)

        return transformer_model


    @classmethod
    def copy_ffn(cls, base_layer, target_layer):
        target_layer.fc1 = base_layer.fc1
        target_layer.fc2 = base_layer.fc2
        return target_layer

#@register_model_architecture("transformer_shared_ffn", "transformer_shared_ffn")
#def transformer_shared(args):
#    base_architecture(args)


@register_model_architecture("transformer_shared_ffn", "transformer_shared_ffn_iwslt_de_en")
def transformer_shared_ffn_iwslt_de_en(args): 
    transformer_iwslt_de_en(args)
