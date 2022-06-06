.PHONY: clean create_environment test_environment requirements download preprocess example_dataset features features_example train train_example predict predict_example eval eval_example

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = {{ cookiecutter.repo_name }}
PYTHON_INTERPRETER = {{ cookiecutter.python_interpreter }}


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files.
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Set up Python virtual environment.
create_environment:
	$(PYTHON_INTERPRETER) -m venv ./venv/
	@echo ">>> Virtual environment created under venv/."

## Test if Python environment is setup correctly.
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Install Python dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

## Download dataset(s) and save in {{cookiecutter.repo_name}}data/raw/.
download: requirements
	$(PYTHON_INTERPRETER) src/data/make_download.py

## Perform preprocessing on raw dataset(s) and save results in {{cookiecutter.repo_name}}data/processed/.
preprocess: download
	$(PYTHON_INTERPRETER) src/data/make_preprocess.py

## Create example dataset from preprocessed data for quicker development.
example_dataset: preprocess
	$(PYTHON_INTERPRETER) src/data/make_example_dataset.py

## Perform feature extraction on the preprocessed data. These features will be the input for the training.
features: preprocess
	$(PYTHON_INTERPRETER) src/data/make_features.py

## (example) Perform feature extraction on the preprocessed data. These features will be the input for the training.
features_example: example_dataset
	$(PYTHON_INTERPRETER) src/data/make_features.py --example

## Do training. The trained models will be store in {{cookiecutter.repo_name}}/models.
train: features
	$(PYTHON_INTERPRETER) src/train/make_train.py

## (example) Do training. The trained models will be store in {{cookiecutter.repo_name}}/models.
train_example: features_example
	$(PYTHON_INTERPRETER) src/train/make_train.py --example

## Make predictions on validation and test sets. The results will be stored in {{cookiecutter.repo_name}}/data.
predict: train
	$(PYTHON_INTERPRETER) src/predict/make_predict.py

## (example) Make predictions on validation and test sets. The results will be stored in {{cookiecutter.repo_name}}/data.
predict_example: train_example
	$(PYTHON_INTERPRETER) src/predict/make_predict.py --example

## Run evaluation script using predictions and ground-truth labels.
eval: predict
	$(PYTHON_INTERPRETER) src/eval/make_eval.py

## (example) Run evaluation script using predictions and ground-truth labels.
eval_example: predict
	$(PYTHON_INTERPRETER) src/eval/make_eval.py --example



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
