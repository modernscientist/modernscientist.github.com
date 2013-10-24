# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Easy Syntax Highlighting With Pygments
# *This notebook first appeared as a [blog](http://themodernscientist.com/posts/2013/2013-10-24-easy_syntax_highlighting_with_pygments/) post on [themodernscientist](http://themodernscientist.com).  
# License: [BSD](http://opensource.org/licenses/BSD-3-Clause) (C) 2013, Michelle L. Gill. Feel free to use, distribute, and modify with the above attribution.*

# <markdowncell>

# As a scientist who splits her time between both wet lab and computational work, I have learned the importance of saving code snippets for future reference. These snippets are handy when I later can't recall a clever Pandas [indexing trick](http://pandas.pydata.org/pandas-docs/stable/indexing.html) or when I want to share code with a colleague.

# <markdowncell>

# I spend most of my time writing code in text editors that have syntax highlighting and I prefer my snippets similarly highlighted. While there are many syntax highlighting engines and websites in existence, one of the most powerful engines happens to be [Pygments](http://pygments.org/), which is python-based, as the name implies.

# <markdowncell>

# Pygments supports an [incredible list](http://pygments.org/languages/) of input languages, called lexers, and has numerous [formatters](http://pygments.org/docs/formatters/) for output. There are also many options available for output [colors and styles](http://pygments.org/docs/styles/). All of these aspects are customizable as well. Lastly, Pygments has few dependencies and can be installed using `easy_install` or MacPorts.
# 
# While Pygments does include many styles, including my favorite, [Monokai](http://www.monokai.nl/blog/2006/07/15/textmate-color-theme), it doesn't have a version of Monokai with a light background. For storing and sharing snippets, I prefer a light background to a dark one. So, I created my own Monokai Light style based on the color scheme [here](https://bitbucket.org/sjl/stevelosh/src/a30885eba5d365da12b264d4beac7596ce1b6ada/media/css/pygments-monokai-light.css?at=default). The style I created can be downloaded directly from [here](https://github.com/mlgill/macports-pygments-monokailight/blob/master/files/monokailight/monokailight.py) and placed in the style directory within Pygments or it can be installed using MacPorts by cloning this [GitHub repository](https://github.com/mlgill/macports-pygments-monokailight).
# 
# Pygments can be run from within a python script or from the terminal using a helper command called `pygmentize`. Both methods are demonstrated for comparison.
# 
# ## Input the Python Snippet
# 
# The code snippet to be highlighted will usually reside in a file or on the clipboard. For the purpose of a self-contained tutorial, it will be input as a string. Since this is predominantly a python blog, it is easy to guess which language is being used.

# <codecell>

python_snippet = r"""
#!/usr/bin/env python

import numpy as np

A = np.random.random((4,4))
Ainv = np.linalg.inv(A)

print 'A ', A
print 'Ainv', Ainv
print 'A*Ainv', np.dot(A,Ainv)"""

# <markdowncell>

# ## Syntax Highlighting in Python
# 
# Pygments is a python module and can be used in the usual fashion to perform syntax highlighting within a python script. This requires the highlighter, lexer, and formatter to be imported. Several formatter options must also be set. First, `full=True` is required to output html with a complete header. Syntax highlighting won't work without this option. Second, the selected style is specified with `style='monokailight'`.

# <codecell>

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
html_snippet = highlight(python_snippet,PythonLexer(),HtmlFormatter(full=True,style='monokailight'))

# <markdowncell>

# The style names of the snippet conflict with those used by this blog, so a regular expression substitution is needed to modify the style names of the snippet. This subsitution should not be necessary in most other use cases.

# <codecell>

import re
html_snippet_sub = re.sub(r"""(body \.|class=")""",r"""\1pygm""",html_snippet)

# <markdowncell>

# Normally, the output would be saved to a file or sent to the clipboard. For this tutorial, it will be displayed using the `HTML` command, which is one of IPython's powerful [display options](http://nbviewer.ipython.org/urls/raw.github.com/ipython/ipython/1.x/examples/notebooks/Part%205%20-%20Rich%20Display%20System.ipynb).

# <codecell>

from IPython.core.display import HTML
HTML(html_snippet_sub)

# <markdowncell>

# ## Highlighting in the Terminal Using Pygmentize
# 
# The most common way I used Pygments is from the terminal via `pygmentize`, which is an accessory command to Pygments. Here is an example with options similar to those used in the python code above:
# 
# ```bash
# pygmentize -l python -f html -O full,style=monokailight \
# -o ./python_snippet.html ./python_snippet.py
# ```
# 
# This command will read the file `python_snippet.py` and output to the file whose name follows the `-o` flag, which is `python_snippet.html` in this case. If the output file is omitted, the html result is returned to the terminal as output. The language (lexer), output format, and options are set with the `-l`, `-f`, and `-O` flags, respectively. The options are `full`, which is analogous to the python `full=True` option and the style selection of monokailight. In many cases, `pygmentize` can guess lexer and output name from the input and output file extensions, so the `-l` and `-f` flags are not always necessary.
# 
# On Mac OS X, code snippets can be copied to the clipboard, sent to `pygmentize`, and then the syntax highlighted result returned to the clipboard using the `pbpaste` and `pbcopy` commands.
# 
# ```bash
# pbpaste | pygmentize -l python -f html -O full,style=monokailight | pbcopy
# ```
# 
# ## Customizing Output: Adding Line Numbers
# 
# There are various options for altering the appearance of the output. Line numbers are one common option, and they can be added with `linenos=True`.

# <codecell>

html_snippet_linenos = highlight(python_snippet,PythonLexer(),HtmlFormatter(full=True,style='monokailight',linenos=True))

# <markdowncell>

# Alternatively, line numbers can be included in `pygmentize` output by adding the option `linenos=1`.
# 
# ```bash
# pygmentize -l python -f html -O full,style=monokailight,linenos=1 \
# -o ./python_snippet_linenos.html ./python_snippet.py
# ```
# 
# Here is the result with line numbers after again making the necessary modifications to avoid conflicts with styles associated with this blog.

# <codecell>

html_snippet_linenos_sub = re.sub(r"""(body \.|class=")""",r"""\1pygm""",html_snippet_linenos)
HTML(html_snippet_linenos_sub)

# <markdowncell>

# That's it! Pygments makes it extremely easy to create syntax highlighted code snippets.

