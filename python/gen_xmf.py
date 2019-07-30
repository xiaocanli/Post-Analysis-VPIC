"""Generate a XDMF meta file for VPIC fields and hydro data
"""
import sys
from lxml import etree

endian = "Little"  # Depending on machine
# dims = "1280 1536 3072"
dims = "640 768 1536"
# dims = "320 384 768"
dims_vector = dims + " 3"
dims_tensor6 = dims + " 6"
dims_tensor9 = dims + " 9"
origin = "-156.25 -187.5 0.0"
# dxdydz = "0.2441406 0.2441406 0.2441406"
dxdydz = "0.48828125 0.48828125 0.48828125"
# dxdydz = "0.9765625 0.9765625 0.9765625"
tstart, tend = 0, 40
nframes = tend - tstart + 1
# tfields = 250.0  # in 1/wpe
tfields = 40.0
tpoints = ""
for i in range(tstart, tend+1):
    tpoints += str(tfields * i) + " "
fields_interval = 2217
# fields_list = [{"name": "E", "vars": ["ex", "ey", "ez"]},
#                {"name": "B", "vars": ["bx", "by", "bz"]},
#                {"name": "j", "vars": ["jx", "jy", "jz"]},
#                {"name": "ve", "vars": ["vex", "vey", "vez"]},
#                {"name": "ue", "vars": ["uex", "uey", "uez"]},
#                {"name": "vi", "vars": ["vix", "viy", "viz"]},
#                {"name": "ui", "vars": ["uix", "uiy", "uiz"]},
#                {"name": "pe", "vars": ["pe-xx", "pe-xy", "pe-xz",
#                                        "pe-yx", "pe-yy", "pe-yz",
#                                        "pe-zx", "pe-zy", "pe-zz"]},
#                {"name": "pi", "vars": ["pi-xx", "pi-xy", "pi-xz",
#                                        "pi-yx", "pi-yy", "pi-yz",
#                                        "pi-zx", "pi-zy", "pi-zz"]},
#                {"name": "asbJ", "vars": ["absJ"]},
#                {"name": "ne", "vars": ["ne"]},
#                {"name": "ni", "vars": ["ni"]}]
# fields_list = [{"name": "E", "vars": ["ex", "ey", "ez"]},
#                {"name": "B", "vars": ["bx", "by", "bz"]},
#                {"name": "j", "vars": ["jx", "jy", "jz"]},
#                {"name": "ue", "vars": ["uex", "uey", "uez"]},
#                {"name": "ui", "vars": ["uix", "uiy", "uiz"]},
#                {"name": "asbJ", "vars": ["absJ"]},
#                {"name": "ne", "vars": ["ne"]},
#                {"name": "ni", "vars": ["ni"]}]
# fields_list = [{"name": "B", "vars": ["bx", "by", "bz"]},
#                {"name": "asbJ", "vars": ["absJ"]}]
fields_list = [{"name": "bx", "vars": ["bx"]},
               {"name": "by", "vars": ["by"]},
               {"name": "bz", "vars": ["bz"]},
               {"name": "ex", "vars": ["ex"]},
               {"name": "ey", "vars": ["ey"]},
               {"name": "ez", "vars": ["ez"]},
               {"name": "jx", "vars": ["jx"]},
               {"name": "jy", "vars": ["jy"]},
               {"name": "jz", "vars": ["jz"]},
               {"name": "uex", "vars": ["uex"]},
               {"name": "uey", "vars": ["uey"]},
               {"name": "uez", "vars": ["uez"]},
               {"name": "uix", "vars": ["uix"]},
               {"name": "uiy", "vars": ["uiy"]},
               {"name": "uiz", "vars": ["uiz"]},
               {"name": "vex", "vars": ["vex"]},
               {"name": "vey", "vars": ["vey"]},
               {"name": "vez", "vars": ["vez"]},
               {"name": "vix", "vars": ["vix"]},
               {"name": "viy", "vars": ["viy"]},
               {"name": "viz", "vars": ["viz"]},
               {"name": "absJ", "vars": ["absJ"]},
               {"name": "ne", "vars": ["ne"]},
               {"name": "ni", "vars": ["ni"]},
               {"name": "vkappa", "vars": ["vkappa"]},
               {"name": "ne_vkappa", "vars": ["ne_vkappa"]}]

xmlns_xi = {'xi': "http://www.w3.org/2001/XInclude"}
root = etree.Element("Xdmf", nsmap=xmlns_xi, Version="2.0")

domain = etree.SubElement(root, "Domain")

topology = etree.SubElement(domain, "Topology",
                            attrib={'name': "topo",
                                    'TopologyType': "3DCoRectMesh",
                                    'Dimensions': dims})
geometry = etree.SubElement(domain, "Geometry",
                            attrib={'name': "geo",
                                    'Type': "ORIGIN_DXDYDZ"})
geometry.append(etree.Comment(" Origin "))
dataitem1 = etree.SubElement(geometry, "DataItem",
                             attrib={'Format': "XML",
                                     'Dimensions': "3"})
dataitem1.text = origin
geometry.append(etree.Comment(" DxDyDz "))
dataitem2 = etree.SubElement(geometry, "DataItem",
                             attrib={'Format': "XML",
                                     'Dimensions': "3"})
dataitem2.text = dxdydz
grid = etree.SubElement(domain, "Grid",
                        attrib={'Name': "TimeSeries",
                                'GridType': "Collection",
                                'CollectionType': "Temporal"})
time = etree.SubElement(grid, "Time", TimeType="HyperSlab")
dataitem = etree.SubElement(time, "DataItem",
                            attrib={'Format': "XML",
                                    'NumberType': "Float",
                                    'Dimensions': str(nframes)})
dataitem.text = tpoints

for tframe in range(tstart, tend+1):
    grid2 = etree.SubElement(grid, "Grid",
                             attrib={'Name': "T" + str(tframe),
                                     'GridType': "Uniform"})
    topo2 = etree.SubElement(grid2, "Topology", Reference="/Xdmf/Domain/Topology[1]")
    geo2 = etree.SubElement(grid2, "Geometry", Reference="/Xdmf/Domain/Geometry[1]")
    for field in fields_list:
        if len(field["vars"]) == 1:
            var_name = field["vars"][0]
            att2 = etree.SubElement(grid2, "Attribute",
                                    attrib={'Name': var_name,
                                            'AttributeType': "Scalar",
                                            'Center': "Node"})
            dataitem = etree.SubElement(att2, "DataItem",
                                        attrib={'Format': "Binary",
                                                'DataType': "Float",
                                                'Precision': "4",
                                                'Endian': endian,
                                                'Dimensions': dims})
            dataitem.text = var_name + "_" + str(tframe * fields_interval) + ".gda"
        if len(field["vars"]) == 3:
            var_name = field["name"]
            att2 = etree.SubElement(grid2, "Attribute",
                                    attrib={'Name': var_name,
                                            'AttributeType': "Vector",
                                            'Center': "Node"})
            vec = etree.SubElement(att2, "DataItem",
                                   attrib={'ItemType': "Function",
                                           'Dimensions': dims_vector,
                                           'Function': "JOIN($0, $1, $2)"})
            for var_name in field["vars"]:
                dataitem = etree.SubElement(vec, "DataItem",
                                            attrib={'Format': "Binary",
                                                    'DataType': "Float",
                                                    'Precision': "4",
                                                    'Endian': endian,
                                                    'Dimensions': dims})
                dataitem.text = var_name + "_" + str(tframe * fields_interval) + ".gda"
        if len(field["vars"]) > 3:
            if len(field["vars"]) == 6:
                atype = "Tensor6"
                dims_data = dims_tensor6
                fun = "JOIN($0 $1 $2 $3 $4 $5)"
            elif len(field["vars"]) == 9:
                atype = "Tensor"
                dims_data = dims_tensor9
                fun = "JOIN($0 $1 $2 $3 $4 $5 $6 $7 $8)"
            var_name = field["name"]
            att2 = etree.SubElement(grid2, "Attribute",
                                    attrib={'Name': var_name,
                                            'AttributeType': atype,
                                            'Center': "Node"})
            vec = etree.SubElement(att2, "DataItem",
                                   attrib={'ItemType': "Function",
                                           'Dimensions': dims_data,
                                           'Function': fun})
            for var_name in field["vars"]:
                dataitem = etree.SubElement(vec, "DataItem",
                                            attrib={'Format': "Binary",
                                                    'DataType': "Float",
                                                    'Precision': "4",
                                                    'Endian': endian,
                                                    'Dimensions': dims})
                dataitem.text = var_name + "_" + str(tframe * fields_interval) + ".gda"

header = '<?xml version="1.0"?>\n'
header += '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'

vpic_xmf = etree.tostring(root, pretty_print=True, encoding='unicode')
vpic_xmf = header + vpic_xmf
with open('./vpic-tracer.xmf', 'wb') as f:
    f.write(vpic_xmf.encode("utf-8"))
