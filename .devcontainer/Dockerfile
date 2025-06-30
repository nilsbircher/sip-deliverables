FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libgtk2.0-0 && \
    apt-get autoremove -yqq --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

USER vscode
