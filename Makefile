run_etl:
	python -m src.main run_etl

run_training:
	python -m src.main run_training

run_barlow_twins:
	PYTHONPATH=. python src/scripts/ssl/run_ssl.py barlow_twins

run_byol:
	PYTHONPATH=. python src/scripts/ssl/run_ssl.py byol

run_finetune:
	PYTHONPATH=. python src/scripts/finetune/run_vae_finetune.py barlow_twins

run_treevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_treevi_training.py

run_pygtreevi:
	PYTHONPATH=. python src/scripts/reconstruction/run_pygtreevi_training.py