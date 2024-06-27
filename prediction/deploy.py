from pathlib import Path

from fastapi import FastAPI

from lithonlp.predict import DrillCoreClassifier

app = FastAPI()

GEOBERTJE = "/projects/y-drive/un_rma/_GEO/Projecten/GEOBertje"

classifier = DrillCoreClassifier.from_directory(
    path=Path(GEOBERTJE, "6_Trained_models/fine_tuned/bundled/best"),
    per_target_model_names=("HL_cor", "NL1_cor", "NL2_cor"),
)


@app.get("/predict")
def drill_core_classes(dutch_description: str):
    return classifier(str(dutch_description), cutoffval=0.1)
