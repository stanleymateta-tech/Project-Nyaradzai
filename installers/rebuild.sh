#!/bin/bash
# Rebuild the .oxt and Word dictionary from dictionaries/sn_ZW.{aff,dic}
set -e
cd "$(dirname "$0")/.."
mkdir -p /tmp/oxt/META-INF /tmp/oxt/dictionaries
cp dictionaries/sn_ZW.aff dictionaries/sn_ZW.dic /tmp/oxt/dictionaries/
cat > /tmp/oxt/META-INF/manifest.xml <<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="http://openoffice.org/2001/manifest">
  <manifest:file-entry manifest:media-type="application/vnd.sun.star.configuration-data"
                       manifest:full-path="dictionaries.xcu"/>
</manifest:manifest>
XML
cat > /tmp/oxt/description.xml <<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<description xmlns="http://openoffice.org/extensions/description/2006"
             xmlns:xlink="http://www.w3.org/1999/xlink">
  <identifier value="org.shonatoolkit.spellcheck.sn_ZW"/>
  <version value="0.2.0"/>
  <display-name><name lang="en">Shona (Zimbabwe) Spellchecker</name></display-name>
  <platform value="all"/>
</description>
XML
cat > /tmp/oxt/dictionaries.xcu <<'XML'
<?xml version="1.0" encoding="UTF-8"?>
<oor:component-data xmlns:oor="http://openoffice.org/2001/registry"
                    xmlns:xs="http://www.w3.org/2001/XMLSchema"
                    oor:name="Linguistic" oor:package="org.openoffice.Office">
  <node oor:name="ServiceManager"><node oor:name="Dictionaries">
    <node oor:name="HunSpellDic_sn_ZW" oor:op="fuse">
      <prop oor:name="Locations" oor:type="oor:string-list">
        <value>%origin%/dictionaries/sn_ZW.aff %origin%/dictionaries/sn_ZW.dic</value>
      </prop>
      <prop oor:name="Format" oor:type="xs:string"><value>DICT_SPELL</value></prop>
      <prop oor:name="Locales" oor:type="oor:string-list"><value>sn-ZW</value></prop>
    </node>
  </node></node>
</oor:component-data>
XML
(cd /tmp/oxt && zip -qXr - META-INF description.xml dictionaries.xcu dictionaries) > installers/shona-spellcheck-0.2.oxt
python3 tools/make_word_dic.py
echo "Installers rebuilt."
