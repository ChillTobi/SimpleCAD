# SimpleCAD
An easy to use open source CAD.
SimpleCAD main idea is to make it's useability as easy as possible. 
Objects are created always with a sketch which can be extruded to different forms and every extrusion itself (exception are rounded shapes) can hold again a sketch.
Tools like: 
- Sketcher (create 2D drawings for extrusion, supported by different reference and constraint options)
- Extrusion (Standard add or remove material perpendicular in sketch direction)
- Ring Extrusion (Circular add or remove material with a given radius around the Sketch or World plane, with helical mode)
- Surface (add surface plane for a sketch)
- Cut mode (place a plane at which an extrusion will be cut for visibility)
- Draft (creates angled shapes)
- Merge (create bridges between objects e.g. flat surface to rounded surface)
- Chamfers
- Fillets
- Gear generator (Spur, Ring, Planetry, Bevel, Rack, Helical) with an additional counter part
- Thread generator (Customize or Standard metric/imperial threads)
- Measurement

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="doc/scad_construction.png">
  <img alt="Logo" src="doc/scad_construction.png">
</picture>

