# import yaml
# from pathlib import Path

# class Config:
#     def __init__(self, config_path="config.yaml"):
#         with open(config_path, "r", encoding="utf-8") as f:
#             self._cfg = yaml.safe_load(f)

#     def get(self, *keys, default=None):
#         node = self._cfg
#         for key in keys:
#             if key not in node:
#                 return default
#             node = node[key]
#         return node

#     # Optional: convenience wrappers
#     @property
#     def audio_files(self):
#         return [Path(p) for p in self.get("paths", "audio_files", default=[])]

#     @property
#     def meeting_id(self):
#         return self.get("paths", "dataset", "meeting_id")

#     @property
#     def words_dir(self):
#         return self.get("paths", "dataset", "words_dir")

#     @property
#     def base_dir(self):
#         return self.get("paths", "dataset", "base_dir")

#     @property
#     def gt_path(self):
#         return self.get("paths", "evaluation", "ground_truth")

#     @property
#     def pred_path(self):
#         return self.get("paths", "evaluation", "prediction")
#     @property
#     def speaker_summary(self):
#         return self.get("paths", "evaluation", "speaker_summary")
#     @property
#     def topic_summary(self):
#         return self.get("paths", "evaluation", "topic_summary")
#     @property
#     def meeting_summary(self):
#         return self.get("paths", "evaluation", "meeting_summary")
    
import yaml
from pathlib import Path


class Config:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        # ==========================================
        # experiment
        # ==========================================

        self.exp_name = self.get(
            "experiment",
            "name"
        )

        # IS1003b.Mix-Headset -> IS1003b
        self._meeting_id = (
            self.exp_name.split(".")[0]
        )

        # ==========================================
        # paths
        # ==========================================

        self.paths = self.get("paths")

    # ==================================================
    # generic getter
    # ==================================================

    def get(self, *keys, default=None):

        node = self._cfg

        for key in keys:

            if key not in node:
                return default

            node = node[key]

        return node

    # ==================================================
    # experiment
    # ==================================================

    @property
    def meeting_id(self):
        return self._meeting_id

    # ==================================================
    # dataset
    # ==================================================

    @property
    def dataset_root(self):
        return Path(
            self.paths["dataset_root"]
        )

    @property
    def base_dir(self):
        return self.dataset_root

    @property
    def words_dir(self):
        return self.dataset_root / "words"

    @property
    def topics_dir(self):
        return self.dataset_root / "topics"

    @property
    def abstractive_dir(self):
        return self.dataset_root / "abstractive"

    @property
    def participant_dir(self):
        return (
            self.dataset_root
            / "participantSummaries"
        )

    # ==================================================
    # audio
    # ==================================================

    @property
    def audio_path(self):
        return (
            Path(self.paths["audio"])
            / f"{self.exp_name}.wav"
        )

    @property
    def audio_files(self):
        return [self.audio_path]

    # ==================================================
    # transcriptions
    # ==================================================

    @property
    def transcription_dir(self):
        return Path(
            self.paths["transcription_dir"]
        )

    @property
    def pred_path(self):
        return (
            self.transcription_dir
            / f"final_transcriptions_{self.exp_name}.json"
        )

    # ==================================================
    # summaries
    # ==================================================

    @property
    def speaker_summary(self):
        return (
            Path(self.paths["speaker_summary_dir"])
            / f"speaker_summarization_{self.exp_name}.json"
        )

    @property
    def topic_summary(self):
        return (
            Path(self.paths["topic_summary_dir"])
            / f"topic_summarization_{self.exp_name}.json"
        )

    @property
    def meeting_summary(self):
        return (
            Path(self.paths["meeting_summary_dir"])
            / f"meeting_summarization_{self.exp_name}.json"
        )

    # ==================================================
    # evaluation
    # ==================================================

    @property
    def evaluation_dir(self):
        return Path(
            self.paths["evaluation_dir"]
        )

    @property
    def gt_path(self):
        return (
            Path("./")
            / f"{self.meeting_id}_words.json"
        )
    
    @property
    def evaluation_output(self):
        return (
            self.evaluation_dir
            / f"evaluation_{self.exp_name}.json"
        )

    # ==================================================
    # pipeline
    # ==================================================

    @property
    def summarization_enabled(self):
        return self.get(
            "pipeline",
            "summarization",
            default=True
        )

    @property
    def diarization_enabled(self):
        return self.get(
            "pipeline",
            "diarization",
            default=True
        )