import bpy
import bmesh
from mathutils import Vector
import math

def run_city_gen_and_extrude():
    # =================================================================================
    # SETUP: Run Add-on and Select Object
    # =================================================================================
    print("--- Starting Script ---")
    
    # 0. Delete default Blender objects (Cube, Light, Camera)
    print("Removing default Blender objects...")
    default_objects = ["Cube", "Light", "Camera"]
    for obj_name in default_objects:
        if obj_name in bpy.data.objects:
            obj_to_delete = bpy.data.objects[obj_name]
            bpy.data.objects.remove(obj_to_delete, do_unlink=True)
            print(f"  Deleted: {obj_name}")
    
    # 1. Run the Add-on
    try:
        bpy.ops.cg.import_node_group()
    except AttributeError:
        print("Error: Command 'bpy.ops.cg.import_node_group' not found.")
        return

    # 1b. Configure City Generator modifier settings
    print("Configuring City Generator modifier settings...")
    try:
        city_gen_obj = bpy.data.objects["City_Generator_2.0_Object"]
        city_gen_modifier = city_gen_obj.modifiers["City_Generator_2.0"]
        city_gen_modifier["Socket_96"] = 0.75
        print("  Socket_96 set to 0.75.")
    except KeyError as e:
        print(f"Warning: Could not configure modifier - {e}")
    except Exception as e:
        print(f"Warning: Unexpected error configuring modifier - {e}")

    # 2. Select the specific object
    target_name = "City_Generator_2.0_Object"
    
    if target_name not in bpy.data.objects:
        print(f"Error: Object '{target_name}' not found.")
        return
        
    obj = bpy.data.objects[target_name]
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    # =================================================================================
    # STEP 1: X-AXIS EXTRUSION
    # Target: (28.491, 0, 0) -> Extrude X 60m (10 times)
    # =================================================================================
    
    # Create BMesh to access geometry
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    
    target_loc_1 = Vector((28.491, 0, 0))
    found_edge_1 = None
    min_dist_1 = 0.5
    
    for edge in bm.edges:
        v1 = edge.verts[0].co
        v2 = edge.verts[1].co
        edge_center = (v1 + v2) / 2
        
        dist = (edge_center - target_loc_1).length
        if dist < min_dist_1:
            min_dist_1 = dist
            found_edge_1 = edge
            
    if found_edge_1:
        found_edge_1.select = True
        bmesh.update_edit_mesh(obj.data)
        
        print(f"Step 1: Found edge at {found_edge_1.verts[0].co}. Extruding X...")

        for i in range(5):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (60, 0, 0)} 
            )
    else:
        print(f"Step 1 Warning: No edge found near {target_loc_1}")

    # =================================================================================
    # STEP 2: Y-AXIS EXTRUSION
    # Target: Select ALL edges at the positive Y boundary (top edge)
    # =================================================================================
    
    # CRITICAL: Free and recreate BMesh because topology changed in Step 1
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.free() 
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    
    # Find the maximum Y coordinate in the mesh
    max_y = max(v.co.y for v in bm.verts)
    tolerance = 0.1
    
    # Select ALL edges where both vertices are at max_y (the top edge)
    found_edges_count = 0
    for edge in bm.edges:
        v1 = edge.verts[0].co
        v2 = edge.verts[1].co
        
        # Check if both vertices are at the maximum Y position
        if abs(v1.y - max_y) < tolerance and abs(v2.y - max_y) < tolerance:
            edge.select = True
            found_edges_count += 1

    if found_edges_count > 0:
        bmesh.update_edit_mesh(obj.data)
        print(f"Step 2: Found {found_edges_count} edge(s) at Y={max_y}. Extruding Y...")
        
        for i in range(5):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (0, 60, 0)} 
            )
            print(f"  > Y-Extrusion {i+1}/10 complete.")
    else:
        print(f"Step 2 Warning: No edges found at maximum Y position")

    # =================================================================================
    # STEP 3: SELECT SPECIFIC FACES AND GENERATE PARK
    # =================================================================================
    
    # 1. Ensure we're in Edit Mode (already done, but confirming)
    # 2. Set Selection Mode to Face Selection
    bpy.ops.mesh.select_mode(type='FACE')
    
    # 3. Clear Selection
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Recreate BMesh to work with updated geometry
    bm.free()
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    # 4. Identify and Select Target Faces
    target_coordinates = [
        Vector((118.49, 178.49, 0)),  # Face 1
        Vector((178.49, 178.49, 0)),  # Face 2
        Vector((118.49, 118.49, 0)),  # Face 3
        Vector((178.49, 118.49, 0))   # Face 4
    ]
    
    tolerance = 0.5  # Margin of error for floating-point variances
    selected_faces_count = 0
    
    print("Step 3: Searching for target faces...")
    for face in bm.faces:
        face_center = face.calc_center_median()
        
        # Check if this face center matches any of our target coordinates
        for target in target_coordinates:
            dist = (face_center - target).length
            if dist <= tolerance:
                face.select = True
                selected_faces_count += 1
                print(f"  Selected face at {face_center}")
                break
    
    if selected_faces_count > 0:
        bmesh.update_edit_mesh(obj.data)
        print(f"Step 3: Selected {selected_faces_count} face(s) for park generation.")
        
        # 5. Apply park attribute
        try:
            bpy.ops.mesh.add_park_attribute(value=1)
            print("Park attribute applied successfully.")
        except AttributeError:
            print("Error: Command 'bpy.ops.mesh.add_park_attribute' not found.")
        except Exception as e:
            print(f"Error applying park attribute: {e}")
    else:
        print("Step 3 Warning: No faces selected for park region. Check coordinates.")
    
    bm.free()

    # =================================================================================
    # STEP 4: CONFIGURE RENDER SETTINGS
    # =================================================================================
    
    print("Step 4: Configuring render settings...")
    
    # Part 1: Switch Render Engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    print("  Render engine set to Cycles.")
    
    # Optional: Set device to GPU if available
    try:
        bpy.context.scene.cycles.device = 'GPU'
        print("  Cycles device set to GPU.")
    except:
        print("  GPU not available, using CPU.")
    
    # Part 2: Set Up World Lighting (Nishita Sky)
    world = bpy.context.scene.world
    
    # Enable nodes for world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    
    # Clear existing nodes
    world_nodes.clear()
    
    # Create Sky Texture node
    sky_texture = world_nodes.new(type='ShaderNodeTexSky')
    sky_texture.location = (-300, 300)
    sky_texture.sky_type = 'NISHITA'  # Set to Nishita sky
    
    # Set sun size to 35 degrees (convert to radians)
    sky_texture.sun_size = math.radians(35)
    print("  Sky Texture node created with Nishita sky type.")
    print("  Sun size set to 35 degrees.")
    
    # Create Background node
    background = world_nodes.new(type='ShaderNodeBackground')
    background.location = (0, 300)
    
    # Create World Output node
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world_output.location = (300, 300)
    
    # Link nodes: Sky Texture -> Background -> World Output
    world_links.new(sky_texture.outputs['Color'], background.inputs['Color'])
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    print("  World lighting configured with Nishita Sky.")

    # =================================================================================
    # STEP 5: CONFIGURE TIMELINE AND SUN ANIMATION
    # =================================================================================
    
    print("Step 5: Configuring timeline and sun animation...")
    
    # Part 1: Increase the Timeline Duration
    bpy.context.scene.frame_end = 1500
    print("  Timeline end frame set to 1500.")
    
    # Part 2a: Set initial keyframe at frame 0
    bpy.context.scene.frame_set(0)
    sky_texture.sun_rotation = 0
    sky_texture.keyframe_insert(data_path="sun_rotation", frame=0)
    
    # Set sun elevation to 0.5 degrees at frame 0 (convert to radians)
    sky_texture.sun_elevation = math.radians(0.5)
    sky_texture.keyframe_insert(data_path="sun_elevation", frame=0)
    print("  Sun rotation set to 0 and keyframed at frame 0.")
    print("  Sun elevation set to 0.5 degrees and keyframed at frame 0.")
    
    # Part 2b: Set keyframe at frame 750 (midpoint - sun elevation peaks at 60 degrees)
    bpy.context.scene.frame_set(750)
    sky_texture.sun_elevation = math.radians(60)
    sky_texture.keyframe_insert(data_path="sun_elevation", frame=750)
    print("  Sun elevation set to 60 degrees and keyframed at frame 750.")
    
    # Part 2c: Navigate to Frame 1500
    bpy.context.scene.frame_set(1500)
    print("  Navigated to frame 1500.")
    
    # Part 3: Set and Keyframe the Sun Rotation and Elevation at frame 1500
    sky_texture.sun_rotation = 2000
    sky_texture.keyframe_insert(data_path="sun_rotation", frame=1500)
    
    # Set sun elevation back to 0.5 degrees at frame 1500 (convert to radians)
    sky_texture.sun_elevation = math.radians(0.5)
    sky_texture.keyframe_insert(data_path="sun_elevation", frame=1500)
    print("  Sun rotation set to 2000 and keyframed at frame 1500.")
    print("  Sun elevation set to 0.5 degrees and keyframed at frame 1500.")

    print("--- Script Finished ---")

# Execute
run_city_gen_and_extrude()