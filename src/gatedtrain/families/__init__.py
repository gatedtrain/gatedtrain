"""Framework families: facts and helpers shared by the adapters of several recipes that run the same framework.

A recipe's adapter imports its family module; the families import nothing from the registry or the adapters (MM-12:
no framework is imported, only its layout is described).
"""
