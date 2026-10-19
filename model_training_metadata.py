"""Mencatat waktu mulai dan selesai training ke manifest.json di dalam bundle."""
from io import BytesIO
import json
from zipfile import ZipFile


def add_training_metadata(bundle, started_at, completed_at):
    """Menambah waktu training (UTC) ke manifest.json tanpa mengubah file model di zip."""
    output = BytesIO()
    with ZipFile(BytesIO(bundle)) as source, ZipFile(output, 'w') as target:
        for entry in source.infolist():
            payload = source.read(entry.filename)
            if entry.filename == 'manifest.json':
                manifest = json.loads(payload)
                manifest['training_started_at'] = started_at
                manifest['training_completed_at'] = completed_at
                payload = json.dumps(manifest, indent=2).encode('utf-8')
            target.writestr(entry, payload)
    return output.getvalue()
