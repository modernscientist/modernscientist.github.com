# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.core.display import HTML

# <codecell>

txt_string = file('./python_snippet.py','r').read()
print txt_string

# <codecell>

!pygmentize -f html -l python -O full,style=monokailight \
-o ./python_snippet.html ./python_snippet.py

# <codecell>

html_text = file('./python_snippet.html','r').read()
HTML(html_text)

# <codecell>

!pygmentize -f html -l python -O full,style=monokailight,linenos=1 \
-o ./python_snippet_linenos.html ./python_snippet.py

# <codecell>

html_text = file('./python_snippet_linenos.html','r').read()
HTML(html_text)

# <codecell>


