{{ (fullname ~ " module") | escape | underline("#")}}

.. automodule:: {{ fullname }}

{% block attributes %}
   {% if attributes %}
      {% if attributes | length is eq 1 %}
{{ _('Module Attribute') | underline("*") }}
      {% else %}
{{ _('Module Attributes') | underline("*") }}

.. autosummary::
   :nosignatures:
         {% for item in attributes %}
   {{ item }}
         {%- endfor %}
      {% endif %}
      {% for item in attributes %}
.. autoattribute:: {{ item }}

      {% endfor %}
   {% endif %}
{% endblock %}

{% block functions %}
   {% if functions %}
      {% if functions | length is eq 1 %}
{{ _('Function') | underline("*") }}
      {% else %}
{{ _('Functions') | underline("*") }}

.. autosummary::
   :nosignatures:
         {% for item in functions %}
   {{ item }}
         {%- endfor %}
      {% endif %}
      {% for item in functions %}
.. autofunction:: {{ item }}

      {% endfor %}
   {% endif %}
{% endblock %}

{% block classes %}
   {% if classes %}
      {% if classes | length is eq 1 %}
{{ _('Class') | underline("*") }}
      {% else %}
{{ _('Classes') | underline("*") }}

.. autosummary::
   :nosignatures:
         {% for item in classes %}
   {{ item }}
         {%- endfor %}
      {% endif %}
      {% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

      {% endfor %}
   {% endif %}
{% endblock %}

{% block exceptions %}
   {% if exceptions %}
      {% if exceptions | length is eq 1 %}
{{ _('Exception') | underline("*") }}
      {% else %}
{{ _('Exceptions') | underline("*") }}

.. autosummary::
   :nosignatures:
         {% for item in exceptions %}
   {{ item }}
         {%- endfor %}
      {% endif %}
      {% for item in exceptions %}
   .. autoexception:: {{ item }}
      :show-inheritance:
         
      {% endfor %}
   {% endif %}
{% endblock %}

{% block modules %}
   {% if modules %}
{{ _('Modules') | underline("*") }} 

.. autosummary::
   :toctree:
   :recursive:
      {% for item in modules %}
   {{ item }}
      {%- endfor %}
   {% endif %}
{% endblock %}
