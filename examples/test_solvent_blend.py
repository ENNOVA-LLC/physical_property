from physical_property import Species, ChemicalComposition, Fluids, SolventBlendSpec

water = Species(name="H2O", amount=1, unit="kg", molar_mass=18.015)
ethanol = Species(name="Ethanol", amount=0, unit="kg", molar_mass=46.07)

comp1 = ChemicalComposition(species=[water], name="H2O")
comp2 = ChemicalComposition(species=[ethanol], name="Ethanol")

fluids = Fluids(compositions=[comp1, comp2])
spec = SolventBlendSpec(base="H2O", solvent="Ethanol", value=[0.5, 1.0], unit="wtf")
new_comps = fluids.from_spec(spec)

for comp in new_comps:
    print(f"Total mass: {comp.total_amount('kg'):.2f} kg")
    for s in comp.species:
        print(f"  {s.name}: {s.amount:.2f} {s.unit}")

print("Done!")
