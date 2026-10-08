"""Rebuild homeshield/templates/index.html from decoded_template.html.

The __bundler/template script tag contains a JSON-encoded HTML string.
Forward-slashes in closing tags must be encoded as \\u002F so the browser's
HTML parser does not see </script> and prematurely close the tag.
"""
import json
import re

with open('decoded_template.html', encoding='utf-8') as f:
    new_html = f.read()

with open('homeshield/templates/index.html', encoding='utf-8') as f:
    bundle = f.read()

tag = '<script type="__bundler/template">'
content_start = bundle.find(tag) + len(tag)

# The JSON string ends with a bare " immediately before \n  </script>
end = bundle.find('"\n  </script>', content_start)
assert end > content_start, 'Could not find end of template JSON block'

print(f'Template block: chars {content_start} to {end + 1}')

# Encode and escape: replace </ with </ so the HTML parser cannot
# see </script> or </head> etc. inside the script tag's raw content.
new_encoded = json.dumps(new_html, ensure_ascii=False)
# Replace with the literal 7-char sequence < \ u 0 0 2 F
new_encoded = new_encoded.replace('</', '<' + '\\' + 'u002F')

# Verify the fix worked
assert '</script>' not in new_encoded, 'Escaping failed: raw </script> still present'
assert 'u002F' in new_encoded, 'Escaping failed: no u002F sequences found'

new_template_block = '\n' + new_encoded
new_bundle = bundle[:content_start] + new_template_block + bundle[end + 1:]

print(f'Old bundle: {len(bundle)}  New bundle: {len(new_bundle)}  Delta: +{len(new_bundle) - len(bundle)}')
print(f'u002F sequences in new block: {new_encoded.count("u002F")}')

with open('homeshield/templates/index.html', 'w', encoding='utf-8') as f:
    f.write(new_bundle)

print('index.html rebuilt OK.')
