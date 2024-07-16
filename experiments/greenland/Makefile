outline := issm/data/geom/IS_outline.exp
mesh := issm/data/geom/IS_mesh.pkl
geom := issm/data/geom/IS_surface.npy
train_exp := expdesign/greenland_train_standard.csv
test_exp := expdesign/greenland_test_standard.csv
moulin_pkl := issm/data/moulins/moulins_catchments.pkl
integrated_melt := issm/data/melt/basin_integrated_inputs.csv
basal_melt := issm/data/melt/IS_friction_melt.npy
velocity := issm/data/velocity/IS_vel.npy

inputs: exp geometry moulins $(integrated_melt) $(velocity) melt

exp: $(train_exp) $(test_exp)

geometry: $(outline) $(mesh) $(geom)

moulins: $(moulin_pkl)

melt: $(basal_melt)

$(train_exp):
	cd expdesign; python -m src.expdesign ../train_config.py > train_exp.out
	cp expdesign/train.dat issm/train/table.dat

$(test_exp):
	cd expdesign; python -m src.expdesign ../test_config.py > test_exp.out
	cp expdesign/test.dat issm/test/table.dat

$(outline):
	cd issm/data/geom; python generate_outline.py > generate_outline.out

$(mesh):
	cd issm/data/geom; python make_mesh.py > make_mesh.out

$(geom):
	cd issm/data/geom; python make_surface_bed.py

$(moulin_pkl):
	cd issm/data/moulins; python place_moulins.py > place_moulins.out

$(integrated_melt):
	cd issm/data/melt; python compute_surface_melt.py > compute_surface_melt.out

$(velocity):
	cd issm/data/velocity; python interp_vel.py > interp_vel.out

$(basal_melt):
	cd issm/data/melt; python compute_basal_melt.py > compute_basal_melt.out

.PHONY:clean
clean:
	rm -f $(mesh)
	rm -f issm/data/geom/IS_bed.npy
	rm -f issm/data/geom/IS_surface.npy
	rm -f issm/data/geom/IS_surface_slope.npy
	rm -f issm/data/geom/*.png
	rm -f issm/data/geom/*.out
	rm -f issm/data/velocity/IS_vel.npy
	rm -f issm/data/velocity/*.out

	rm -f expdesign/*
	rm -f issm/train/*.dat
	rm -f issm/test/*.dat

	rm -f $(integrated_melt)
	rm -f $(basal_melt)
	rm -f issm/data/melt/*png
	rm -f issm/data/melt/*.out

	rm -f $(moulin_pkl)
	rm -f issm/data/moulins/*png
	rm -f issm/data/moulins/*.out
	rm -f issm/data/moulins/moulin_indices.csv

