set dotenv-load := true
set dotenv-filename := ".env"

root := justfile_directory()
image := "news2etf-engine:latest"

local-deploy:
    docker compose up -d loki grafana

build:
    docker build --network host -t {{image}} .

run:
    docker run --rm -d \
        --user "$(id -u):$(id -g)" \
        --env-file .env \
        --network host \
        --volume {{root}}/config.toml:/app/config.toml:ro \
        --volume {{root}}/data:/app/data \
        {{image}} \
        python main.py labeling label --sample 5000 --level 1 -f

build-dict:
    uv run main.py industry build-dict

label-level1:
    uv run main.py labeling label --sample 100 --level 1 -f

label-level2:
    uv run main.py labeling label --sample 50000 --level 2