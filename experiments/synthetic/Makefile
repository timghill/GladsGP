mesh := issm/data/geom/synthetic_mesh.pkl
geom := issm/data/geom/synthetic_surface.npy
train_exp := expdesign/synthetic_train_standard.csv
test_exp := expdesign/synthetic_test_standard.csv
moulin_pkl := issm/data/moulins/moulins_catchments.pkl
integrated_melt := issm/data/melt/basin_integrated_inputs.csv

inputs: exp geometry moulins $(integrated_melt)

exp: $(train_exp) $(test_exp)

geometry: $(mesh) $(geom)

moulins: $(moulin_pkl)

$(train_exp):
	cd expdesign; python -m src.expdesign ../train_config.py > train_exp.out
	cp expdesign/train.dat issm/train/table.dat

$(test_exp):
	cd expdesign; python -m src.expdesign ../test_config.py > test_exp.out
	cp expdesign/test.dat issm/test/table.dat

$(mesh):
	cd issm/data/geom; python make_mesh.py > make_mesh.out

$(geom):
	cd issm/data/geom; python make_surface_bed.py

$(moulin_pkl):
	cd issm/data/moulins; python place_moulins.py > place_moulins.out

$(integrated_melt):
	cd issm/data/melt; python compute_melt.py > compute_melt.out

.PHONY:clean
clean:
	rm -f $(mesh)
	rm -f issm/data/geom/synthetic_surface.npy
	rm -f issm/data/geom/synthetic_bed.npy
	rm -f issm/data/geom/*.png

	rm -f expdesign/*
	rm -f issm/train/*.dat
	rm -f issm/test/*.dat

	rm -f $(integrated_melt)
	rm -f issm/data/melt/*png

	rm -f $(moulin_pkl)
	rm -f issm/data/moulins/*png

