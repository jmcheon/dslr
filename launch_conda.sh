#!/bin/sh

# Function to determine the appropriate Miniconda script based on the OS
which_dl() {
    # If operating system name contains Darwin: MacOS. Else Linux
    if uname -s | grep -iqF Darwin; then
        echo "Miniconda3-latest-MacOSX-x86_64.sh"
    else
        echo "Miniconda3-latest-Linux-x86_64.sh"
    fi
}

# Function to determine the current shell
which_shell() {
    # Check if the $SHELL variable contains "zsh", if yes, then zsh, else Bash
    if echo "$SHELL" | grep -iqF zsh; then
        echo "zsh"
    else
        echo "bash"
    fi
}

# Function to check and install 42AI environment
when_conda_exist() {
    printf "Checking 42AI-$USER environment: "
    if conda info --envs | grep -iqF "42AI-$USER"; then
        printf "\e[33mDONE\e[0m\n"
    else
        printf "\e[31mKO\e[0m\n"
        printf "\e[33mCreating 42AI environment:\e[0m\n"
        conda update -n base -c defaults conda -y
        conda create --name "42AI-$USER" python=3.11 $REQUIREMENTS
    fi
}

# Function to set up conda
set_conda() {
    # Define Miniconda path based on the OS
    if [ -d "/goinfre" ]; then
        MINICONDA_PATH="/goinfre/$USER/miniconda3"
    else
        MINICONDA_PATH="/home/$USER/miniconda3"
    fi
    CONDA="$MINICONDA_PATH/bin/conda"
    PYTHON_PATH=$(which python)
    SCRIPT=$(which_dl)
    MY_SHELL=$(which_shell)
    DL_LINK="https://repo.anaconda.com/miniconda/$SCRIPT"
    DL_LOCATION="/tmp/"
    printf "Checking conda: "
    # Check if conda is available
    if command -v conda >/dev/null 2>&1; then
        printf "\e[32mOK\e[0m\n"
        when_conda_exist
        printf "\e[33mLaunch the following command or restart your shell:\e[0m\n"
        if [ "$MY_SHELL" = "zsh" ]; then
            printf "\tsource ~/.zshrc\n"
        else
            printf "\tsource ~/.bashrc\n"
        fi
        return
    fi
    printf "\e[31mKO\e[0m\n"
    if [ ! -f "$DL_LOCATION$SCRIPT" ]; then
        printf "\e[33mDownloading installer:\e[0m\n"
        curl -Lo "$DL_LOCATION$SCRIPT" "$DL_LINK"
    fi
    printf "\e[33mInstalling conda:\e[0m\n"
    sh "$DL_LOCATION$SCRIPT" -b -p "$MINICONDA_PATH"
    printf "\e[33mConda initial setup:\e[0m\n"
    "$CONDA" init "$MY_SHELL"
    "$CONDA" config --set auto_activate_base false
    printf "\e[33mCreating 42AI-$USER environment:\e[0m\n"
    "$CONDA" update -n base -c defaults conda -y
    "$CONDA" create --name "42AI-$USER" python=3.11 $REQUIREMENTS 
    printf "\e[33mLaunch the following command or restart your shell:\e[0m\n"
    if [ "$MY_SHELL" = "zsh" ]; then
        printf "\tsource ~/.zshrc\n"
    else
        printf "\tsource ~/.bashrc\n"
    fi
}

# Packages to be installed
REQUIREMENTS="jupyter numpy pandas pycodestyle matplotlib seaborn scikit-learn -y"

# Execute the setup
set_conda

