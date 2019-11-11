from typing import Tuple, List

import logging
from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

log = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register('relex')
class RelationExtractionPredictor(Predictor):
    """"Predictor wrapper for the RelationExtractionPredictor"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        e1 = json_dict['e1']
        e2 = json_dict['e2']
        mentions = json_dict['mentions']

        instance = self._dataset_reader.text_to_instance(
                e1=e1, e2=e2, rels=[], mentions=mentions, is_predict=True, is_supervised_bag=False)
        if not instance:
            log.error('parsing instance failed: %s', mentions)
            instance = self._dataset_reader.text_to_instance(
                    e1="e1", e2="e2", rels=[],
                    mentions=["Some relation between <e1>entity 1</e1> and <e2>entity 2</e2>"],
                    is_predict=True, is_supervised_bag=False)
        return instance, {}

    # @overrides
    # def predict_json(self, inputs: JsonDict) -> JsonDict:
    #     instance = self._json_to_instance(inputs)
    #     result = self.predict_instance(instance)
    #     result['entities'] = [inputs['e1'], inputs['e2']]
    #     result['mentions'] = inputs['mentions']
    #
    #     return result

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        out = self._model.forward_on_instance(instance)
        out['entities'] = list(instance['metadata']['entities'])
        out['mentions'] = list(instance['metadata']['mentions'])

        return sanitize(out)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for out, instance in zip(outputs, instances):
            out['entities'] = list(instance['metadata']['entities'])
            out['mentions'] = list(instance['metadata']['mentions'])
        return sanitize(outputs)