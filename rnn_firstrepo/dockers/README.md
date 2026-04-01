Questa cartella contiene dockerfile e docker-compose.yml conservati tutti insieme. Quando devo lanciare dei job su macchine esterne utilizzo quindi gli ambienti che costruisco.

Qui tutto l'ho preparato per lanciarlo su un'istanza di OKD, di Openshift.

Le strutture cartelle, nomi file e qualsiasi altra cosa va preparata, qui non c'è roba pronta per essere buildata.

## IMDB

### Preparazione progetto

Il progetto deve mostrare la seguente struttura prima del push su un registry di immagini

```
progetto/
├── train.py
├── pyproject.toml
├── poetry.lock          ← genera con: poetry lock
├── Dockerfile
├── docker-compose.yml
└── output/              
    ├── models/          ← salva qui il modello da train.py
    └── logs/            ← log con tutti i print, uno per run
```

Lanciare i seguenti comandi per caricare du DockerHub l'immagine dopo averla costruita:

```
docker compose build
docker compose up
docker tag rnn-firstrepo:jongonns/rnn-imdb-test:latest
docker push jongonns/rnn-imdb-test:latest
```

### Lancio job su OKD

1. Creare un PVC: Menu laterale → Storage → PersistentVolumeClaims → Create PersistentVolumeClaim. Usare il nome assegnato qui nello yaml del job.

2. Vai su Workload/Jobs e creane uno: qui inserisci il contenuto del file `imdb_OKD-job-config.yml`

3. Dal pod controlla gli Events ed il Log per vedere che sia tutto okay

## Chars