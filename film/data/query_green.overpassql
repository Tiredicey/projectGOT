[out:json][timeout:170];
(
 relation["natural"="water"]["name"~"Taal",i](13.80,120.85,14.15,121.25);
 way["natural"="water"](13.877,121.03,14.06,121.26);
 way["landuse"~"^(forest|orchard|farmland|meadow|grass|plant_nursery)$"](13.877,121.03,14.06,121.26);
 way["natural"~"^(wood|scrub|grassland|tree_row)$"](13.877,121.03,14.06,121.26);
 relation["landuse"~"^(forest|orchard)$"](13.877,121.03,14.06,121.26);
 relation["natural"="wood"](13.877,121.03,14.06,121.26);
);
out geom;
