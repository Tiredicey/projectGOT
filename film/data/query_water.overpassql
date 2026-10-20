[out:json][timeout:150];
(
 relation["boundary"="administrative"]["name"="Lipa"];
 way["natural"="water"](13.90,121.08,14.03,121.20);
 way["waterway"~"river|stream"](13.90,121.08,14.03,121.20);
 node["natural"="peak"](13.85,121.05,14.10,121.30);
);
out geom;
