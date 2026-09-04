"""
Review app: correct a batch analysis and finalise it.

Separate from sem_analysis_app, which analyses images from scratch. This one
never segments — it opens masks somebody else made and lets them be corrected.
The two share their state, their refinement tools and their results file format,
so a folder can move between them.
"""
