import bpy

def create_rgb_shader_material(name, color_val):
    """
    Creates a material with an RGB node connected directly to the Output.
    color_val: tuple (R, G, B, A)
    """
    # Check if material exists, else create it
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name=name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create RGB Node
    node_rgb = nodes.new(type='ShaderNodeRGB')
    node_rgb.location = (-200, 0)
    node_rgb.outputs[0].default_value = color_val
    
    # Create Output Node
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (0, 0)
    
    # Link RGB directly to Surface Output
    links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
    
    return mat

def apply_material_recursive(collection, material):
    """
    Recursively applies the given material to all meshes in this collection 
    and all of its child collections.
    """
    # 1. Apply to objects in the current collection
    for obj in collection.objects:
        if obj.type == 'MESH':
            # Clear existing materials and append the new one
            obj.data.materials.clear()
            obj.data.materials.append(material)
            
    # 2. Recursively apply to child collections
    for child_col in collection.children:
        apply_material_recursive(child_col, material)

def main():
    # --- CONFIGURATION ---
    # We define distinct grayscale colors for your categories
    # Format: (R, G, B, A)
    mat_buildings = create_rgb_shader_material("Mat_Override_Buildings", (0.8, 0.8, 0.8, 1.0)) # Light Grey
    mat_cars      = create_rgb_shader_material("Mat_Override_Cars",      (0.2, 0.2, 0.2, 1.0)) # Dark Grey
    mat_sidewalks = create_rgb_shader_material("Mat_Override_Sidewalks", (0.5, 0.5, 0.5, 1.0)) # Mid Grey
    mat_signs     = create_rgb_shader_material("Mat_Override_Signs",     (1.0, 1.0, 1.0, 1.0)) # White
    mat_trees     = create_rgb_shader_material("Mat_Override_Trees",     (0.3, 0.3, 0.3, 1.0)) # Dark-Mid Grey
    mat_lights    = create_rgb_shader_material("Mat_Override_Lights",    (0.9, 0.9, 0.9, 1.0)) # Very Light Grey
    mat_streets   = create_rgb_shader_material("Mat_Override_Streets",   (0.1, 0.1, 0.1, 1.0)) # Very Dark Grey
    mat_grass     = create_rgb_shader_material("Mat_Override_Grass",     (0.4, 0.4, 0.4, 1.0)) # Mid-Dark Grey

    # --- UNLINK DECALS COLLECTION FROM MODIFIER ---
    print("Unlinking decals collection from City Generator modifier...")
    try:
        city_gen_obj = bpy.data.objects["City_Generator_2.0_Object"]
        city_gen_modifier = city_gen_obj.modifiers["City_Generator_2.0"]
        # Socket_26 is the Decal Collection input
        city_gen_modifier["Socket_26"] = None
        print("Decals collection unlinked successfully.")
    except KeyError as e:
        print(f"Warning: Could not unlink decals collection - {e}")
    except Exception as e:
        print(f"Warning: Unexpected error unlinking decals - {e}")

    # --- TRAVERSAL ---
    root_name = "City_Gen_2.0"
    
    # Ensure the root collection exists
    if root_name not in bpy.data.collections:
        print(f"Error: Collection '{root_name}' not found.")
        return
        
    root_col = bpy.data.collections[root_name]
    
    # Navigate to City_Gen_2.0_Assets collection
    assets_col = None
    if "City_Gen_2.0_Assets" in root_col.children:
        assets_col = root_col.children["City_Gen_2.0_Assets"]
    else:
        print("Warning: 'City_Gen_2.0_Assets' collection not found. Trying to use root collection directly.")
        assets_col = root_col
    
    # 1. Process "01 Building Assets"
    # We look for this specific collection inside the assets collection
    if "01 Building Assets" in assets_col.children:
        print("Processing Buildings...")
        building_col = assets_col.children["01 Building Assets"]
        apply_material_recursive(building_col, mat_buildings)
    else:
        print("Warning: '01 Building Assets' collection not found.")

    # 2. Process "02 Street Assets"
    if "02 Street Assets" in assets_col.children:
        street_col = assets_col.children["02 Street Assets"]
        
        # Inside Street Assets, we look for specific sub-groups
        
        # A. Car Assets
        if "Car Assets" in street_col.children:
            print("Processing Cars...")
            apply_material_recursive(street_col.children["Car Assets"], mat_cars)
        else:
            print("Warning: 'Car Assets' collection not found.")
            
        # B. Side Walk Assets (with special handling for specific sub-collections)
        if "Side Walk Assets" in street_col.children:
            sidewalk_col = street_col.children["Side Walk Assets"]
            
            # Process trees separately
            if "trees" in sidewalk_col.children:
                print("Processing Trees...")
                apply_material_recursive(sidewalk_col.children["trees"], mat_trees)
            else:
                print("Warning: 'trees' collection not found.")
            
            # Process Scaffolding, Street Lights, and traffic lights together
            for light_col_name in ["Scaffolding", "Street Lights", "traffic lights"]:
                if light_col_name in sidewalk_col.children:
                    print(f"Processing {light_col_name}...")
                    apply_material_recursive(sidewalk_col.children[light_col_name], mat_lights)
                else:
                    print(f"Warning: '{light_col_name}' collection not found.")
            
            # Process remaining sidewalk assets (excluding trees, Scaffolding, Street Lights, traffic lights)
            print("Processing remaining Sidewalk Assets...")
            excluded_collections = {"trees", "Scaffolding", "Street Lights", "traffic lights"}
            for child_col in sidewalk_col.children:
                if child_col.name not in excluded_collections:
                    apply_material_recursive(child_col, mat_sidewalks)
            
            # Also apply to direct objects in Side Walk Assets collection
            for obj in sidewalk_col.objects:
                if obj.type == 'MESH':
                    obj.data.materials.clear()
                    obj.data.materials.append(mat_sidewalks)
        else:
            print("Warning: 'Side Walk Assets' collection not found.")
            
        # C. traffic signs (Note case sensitivity matches your folder structure)
        if "traffic signs" in street_col.children:
            print("Processing Traffic Signs...")
            apply_material_recursive(street_col.children["traffic signs"], mat_signs)
        else:
            print("Warning: 'traffic signs' collection not found.")
            
    else:
        print("Warning: '02 Street Assets' collection not found.")
    
    # 3. Process CityGen_Streets material (not in collections, but exists as material)
    print("Processing CityGen_Streets material...")
    if "CityGen_Streets" in bpy.data.materials:
        citygen_streets_mat = bpy.data.materials["CityGen_Streets"]
        # Replace its shader nodes with RGB material
        citygen_streets_mat.use_nodes = True
        nodes = citygen_streets_mat.node_tree.nodes
        links = citygen_streets_mat.node_tree.links
        nodes.clear()
        
        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.location = (-200, 0)
        node_rgb.outputs[0].default_value = (0.1, 0.1, 0.1, 1.0)  # Very Dark Grey
        
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_out.location = (0, 0)
        
        links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
        print("CityGen_Streets material updated.")
    else:
        print("Warning: 'CityGen_Streets' material not found.")
    
    # 3b. Process CityGenside walks material (same as Side Walk Assets)
    print("Processing CityGenside walks material...")
    if "CityGenside walks" in bpy.data.materials:
        citygen_sidewalks_mat = bpy.data.materials["CityGenside walks"]
        # Replace its shader nodes with RGB material
        citygen_sidewalks_mat.use_nodes = True
        nodes = citygen_sidewalks_mat.node_tree.nodes
        links = citygen_sidewalks_mat.node_tree.links
        nodes.clear()
        
        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.location = (-200, 0)
        node_rgb.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)  # Mid Grey (same as sidewalks)
        
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_out.location = (0, 0)
        
        links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
        print("CityGenside walks material updated.")
    else:
        print("Warning: 'CityGenside walks' material not found.")
    
    # 3c. Process CityGenroof materials (same as Building Assets)
    for roof_mat_name in ["CityGenroof.001", "CityGenroof.002", "CityGenroof.003"]:
        print(f"Processing {roof_mat_name} material...")
        if roof_mat_name in bpy.data.materials:
            citygen_roof_mat = bpy.data.materials[roof_mat_name]
            # Replace its shader nodes with RGB material
            citygen_roof_mat.use_nodes = True
            nodes = citygen_roof_mat.node_tree.nodes
            links = citygen_roof_mat.node_tree.links
            nodes.clear()
            
            node_rgb = nodes.new(type='ShaderNodeRGB')
            node_rgb.location = (-200, 0)
            node_rgb.outputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Light Grey (same as buildings)
            
            node_out = nodes.new(type='ShaderNodeOutputMaterial')
            node_out.location = (0, 0)
            
            links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
            print(f"{roof_mat_name} material updated.")
        else:
            print(f"Warning: '{roof_mat_name}' material not found.")
    
    # 3d. Process CityGen_Grass material
    print("Processing CityGen_Grass material...")
    if "CityGen_Grass" in bpy.data.materials:
        citygen_grass_mat = bpy.data.materials["CityGen_Grass"]
        # Replace its shader nodes with RGB material
        citygen_grass_mat.use_nodes = True
        nodes = citygen_grass_mat.node_tree.nodes
        links = citygen_grass_mat.node_tree.links
        nodes.clear()
        
        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.location = (-200, 0)
        node_rgb.outputs[0].default_value = (0.4, 0.4, 0.4, 1.0)  # Mid-Dark Grey
        
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_out.location = (0, 0)
        
        links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
        print("CityGen_Grass material updated.")
    else:
        print("Warning: 'CityGen_Grass' material not found.")
    
    # 3e. Process lane materials (same as streets - very dark grey)
    lane_materials = ["CityGen_lanes_secondary_color", "CityGen_lanes_white", "CityGenbasic dark green"]
    for lane_mat_name in lane_materials:
        print(f"Processing {lane_mat_name} material...")
        if lane_mat_name in bpy.data.materials:
            lane_mat = bpy.data.materials[lane_mat_name]
            # Replace its shader nodes with RGB material
            lane_mat.use_nodes = True
            nodes = lane_mat.node_tree.nodes
            links = lane_mat.node_tree.links
            nodes.clear()
            
            node_rgb = nodes.new(type='ShaderNodeRGB')
            node_rgb.location = (-200, 0)
            node_rgb.outputs[0].default_value = (0.1, 0.1, 0.1, 1.0)  # Very Dark Grey (same as streets)
            
            node_out = nodes.new(type='ShaderNodeOutputMaterial')
            node_out.location = (0, 0)
            
            links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
            print(f"{lane_mat_name} material updated.")
        else:
            print(f"Warning: '{lane_mat_name}' material not found.")
    
    # 3f. Process CityGen_Curb material (same as sidewalks)
    print("Processing CityGen_Curb material...")
    if "CityGen_Curb" in bpy.data.materials:
        citygen_curb_mat = bpy.data.materials["CityGen_Curb"]
        # Replace its shader nodes with RGB material
        citygen_curb_mat.use_nodes = True
        nodes = citygen_curb_mat.node_tree.nodes
        links = citygen_curb_mat.node_tree.links
        nodes.clear()
        
        node_rgb = nodes.new(type='ShaderNodeRGB')
        node_rgb.location = (-200, 0)
        node_rgb.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)  # Mid Grey (same as sidewalks)
        
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_out.location = (0, 0)
        
        links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
        print("CityGen_Curb material updated.")
    else:
        print("Warning: 'CityGen_Curb' material not found.")
    
    # 4. Process Roof_Materials node group
    print("Processing Roof_Materials...")
    if "Roof_Materials" in bpy.data.node_groups:
        roof_node_group = bpy.data.node_groups["Roof_Materials"]
        # Find all materials that use this node group and update them
        for mat in bpy.data.materials:
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'GROUP' and node.node_tree == roof_node_group:
                        # This material uses Roof_Materials, convert it to buildings material
                        print(f"  Updating material '{mat.name}' that uses Roof_Materials...")
                        mat.node_tree.nodes.clear()
                        
                        node_rgb = mat.node_tree.nodes.new(type='ShaderNodeRGB')
                        node_rgb.location = (-200, 0)
                        node_rgb.outputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Same as buildings
                        
                        node_out = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
                        node_out.location = (0, 0)
                        
                        mat.node_tree.links.new(node_rgb.outputs['Color'], node_out.inputs['Surface'])
                        break  # Only need to check once per material
    else:
        print("Warning: 'Roof_Materials' node group not found.")
        
    print("Script Finished.")

# Execute
main()