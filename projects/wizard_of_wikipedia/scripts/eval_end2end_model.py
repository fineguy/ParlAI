#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.scripts.eval_model import eval_model
from parlai.zoo.wizard_of_wikipedia.full_dialogue_retrieval_model import download
from projects.wizard_of_wikipedia.generator.agents import (
    EndToEndAgent,
)


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    EndToEndAgent.add_cmdline_args(parser)

    parser.set_params(
        task='wizard_of_wikipedia:generator:random_split',
        model='projects.wizard_of_wikipedia.generator.agents:EndToEndAgent',
        model_file='models:wizard_of_wikipedia/end2end_generator/model',
        dict_lower=True,
        dict_tokenizer='bpe',
        n_layers=5,
        n_heads=2,
        dropout=0.20,
        ffn_size=512,
        embedding_size=256,
        log_every_n_secs=10,
        validation_patience=12,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_every_n_epochs=0.5,
        n_positions=128,
        truncate=128,
        max_knowledge=32,
        knowledge_alpha=0.95,
        knowledge_truncate=32,
        learningrate=5e-4,
        warmup_updates=5000,
        clip=0.1,
        lr_scheduler='invsqrt',
        embedding_type='fasttext',
        beam_size=1,
        skip_generation=False,
        batchsize=64,
    )

    opt = parser.parse_args()
    download(opt['datapath'])  # download pretrained end2end model

    eval_model(opt)
