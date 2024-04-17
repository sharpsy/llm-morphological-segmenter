FROM python:3.10

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,target=/tmp/requirements.txt,source=requirements.txt \
    pip install --root-user-action=ignore --disable-pip-version-check -r /tmp/requirements.txt

