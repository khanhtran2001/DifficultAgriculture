from dagri.interfaces import ScorerInterface, ScoringConfig, ScoringResults
from dagri.scoring.min_scorer import MinScorer

class Scorer(ScorerInterface):
    def __init__(self, scoring_config: ScoringConfig):
        scorer_type = getattr(scoring_config, "type", "min_scorer")
        if scorer_type != "min_scorer":
            raise ValueError(f"Unsupported scorer type: {scoring_config.type}")
        self.scorer = MinScorer(scoring_config)

    def score(self, optimal_conf_threshold_prediction_dir: str, low_conf_prediction_dir: str, dataset_properties) -> ScoringResults:
        return self.scorer.score(optimal_conf_threshold_prediction_dir, low_conf_prediction_dir, dataset_properties)