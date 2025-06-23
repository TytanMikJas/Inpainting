run_etl:
	python -m src.main run_etl

run_training:
	python -m src.main run_training

run_barlow_twins:
	PYTHONPATH=. python src/scripts/ssl/run_ssl.py barlow_twins

run_byol:
	PYTHONPATH=. python src/scripts/ssl/run_ssl.py byol

run_barlow_twins_finetune:
	PYTHONPATH=. python src/scripts/finetune/run_vae_finetune.py barlow_twins

run_byol_finetune:
	PYTHONPATH=. python src/scripts/finetune/run_vae_finetune.py byol

run_simclr:
	PYTHONPATH=. python  src/scripts/ssl/run_ssl.py simclr

run_simclr_finetune:
	PYTHONPATH=. python src/scripts/finetune/run_simclr_finetune.py


run_treevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_treevi_training.py

run_pygtreevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_pygtreevi_training.py

run_vqvae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vqvae_training.py

run_vae:
	PYTHONPATH=. python src/scripts/reconstruction/run_vae_training.py

jupyter:
	uv run --with jupyter --active jupyter lab

reproduce: run_etl run_barlow_twins run_byol run_barlow_twins_finetune run_byol_finetune  run_treevi run_pygtreevi run_vqvae run_vae
