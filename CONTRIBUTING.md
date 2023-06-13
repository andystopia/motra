# Contribution Documentation
MOTRA is open for contributions. 

MOTRA is written in Python –– one of the most popular languages for data science. 

For tracking dependencies, we happen to have chosen [poetry](https://python-poetry.org/), as our dependency manager. To really integrate well with this software. You will need to [install poetry](https://python-poetry.org/docs/#installation). 

You will also need [Python 3.9](https://www.python.org/downloads/release/python-390/). 

## Pre-Requisite Installs
 - [Python 3.9](https://www.python.org/downloads/release/python-390/). 
 - [Python Poetry](https://python-poetry.org/)
 - [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (optional but highly recommended)

## Getting Started

To begin either go to github, and click the green `Code`, followed by `Download Zip`, or more elegantly

```sh
git clone github.com/andystopia/motra
```

Consider forking the project first so that you can contribute back upstream with PRs (but this can always be done afterwards, so don't sweat it!)

If you successfully complete these steps, you're now ready to start developing motra. Open your terminal in the directory where you placed motra (or for those using PyCharm/IDEA/VSCode, just open the directory, and open the terminal in the editor).

## Protect Your Builds with Poetry

Poetry makes sure that you will *and* the next person will be able to build whatever motra has got going on! Please use it! Don't reach for `pip`, don't type commands that include `pip`, they *are almost never* what you're going to do what you want. We've all struggled to build software before. Please use poetry. 

Starting out. In your terminal session you have open. Type 

```sh
poetry shell
```

There may be an error saying env is not defined or something along those lines. That's a problem, that can be fixed with 

```sh
poetry env use 3.9
```

Assuming you have python 3.9 installed, as specified.
If this succeeds, retype 
```
poetry shell
```
Not much will look like it's happened, but if it's quiet that's okay! It's set up an development environment for you, which you can check by typing `which python`. If it's a long, weird path, you're on the right track!

**WARNING**: for those of you who have used anaconda, conda, miniconda, etc, deactivate your venv, by spamming `conda deactivate` followed by return in the terminal a couple of times. It will screw with your python installation if you don't.

Now that the shell is open, type poetry install. It will install the versions of packages in the lockfile, thereby ensuring that everyone uses the same version of packages. We are a science lab, after-all, sometimes libraries have bugs, and we want to know if we're affected. 

You know have a python and `jupyter-lab` installation up and going! You can start a jupyter server with `jupyter-lab` or you can run python as normal (it's named `python`). 


## Guidelines

Please docstring any user intended functions. Documentation is important. Maybe one day I'll get a Github Pages Documentation Page going. 


## Words of Warning.

### That  run button.

That VSCode/IDEA/etc run button is probably the devil. It probably does not integrate with poetry. This isn't because poetry is bad or the tools don't have support for poetry, it's probably just because it isn't *configured*. You could configure it. Please configure it if you want to use it.

Motra is primarily a library anyways, so you're probably going to be in notebooks.

### VSCode-Jupyter Doesn't Work On Silicon

Yep. Broken. Pip and VSCode disagree over which dynamic library architecture to load. Couldn't solve it myself. Please let me know if you do. Wasn't worth my time. 

