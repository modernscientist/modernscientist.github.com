FROM andrewosh/binder-base

MAINTAINER Michelle Gill <michelle@michellelynngill.com>

USER main

# Add my conda channel and install other dependencies
RUN echo "channels:\n  - mlgill" >> ~/.condarc
RUN conda install -c mlgill jsanimation cvxopt_glpk pygments-style-monokailight