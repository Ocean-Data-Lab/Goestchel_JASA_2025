# Define the interpreter
PYTHON = python

# Define the source files
SRC2 = main_section2.py
SRC3 = main_section3.py
SRC4 = main_section4.py
SRC4c = main_section4c.py
SRC5 = main_section5.py
SRCSUPP = main_supplementary.py

# Define the virtual environment directory
VENV_DIR = venv

# Check and activate virtual environment if necessary
ACTIVATE_VENV = \
    if [ -z "$$VIRTUAL_ENV" ]; then \
        . $(VENV_DIR)/bin/activate; \
    fi

# Start the compilation (execution in this case)
all: section2 section3 section4 section5

section2:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC2)

section3:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC3)

section4:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC4)

section4c:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC4c)

section5:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC5)

supplementary:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRCSUPP)