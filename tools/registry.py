from . import dft, literature, materials_project, wannier90, phonon, qe2pert

TOOLS = {
   "literature_search": (literature.literature_search, literature.LiteratureSearchInput),
   "query_materials_project":(materials_project.query_materials_project, materials_project.MPQueryInput),
   "run_dft":(dft.run_dft, dft.DFTInput),
   "run_wannier90":(wannier90.run_wannier90, wannier90.Wannier90Input),
   "create_ph_save":(phonon.create_ph_save, phonon.PhononInput),
   "run_qe2pert":(qe2pert.run_qe2pert, qe2pert.Qe2pertInput),
}

TOOL_SPECS = [literature.TOOL_SPEC, materials_project.TOOL_SPEC, dft.TOOL_SPEC, wannier90.TOOL_SPEC, phonon.TOOL_SPEC, qe2pert.TOOL_SPEC]

def dispatch(name: str, raw_input: dict)-> dict | list | str:
   """Validate input against the Pydantic schema, then call the tool."""
   if name not in TOOLS:
      return {"error":f"Unknown tool '{name}'. Available: {list(TOOLS)}"}
   func, schema = TOOLS[name]
   try:
      validated = schema(**raw_input)
   except Exception as e:
      # Hand the validation error back to the agent - it will retry.
      return {"error": f"Invalid input for {name}:{e}"}
   try:
      return func(**validated.model_dump())
   except Exception as e:
      return {"error": f"Tool '{name} raised: {type(e).__name__}:{e}'"}
   