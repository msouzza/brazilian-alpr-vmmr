"""
Módulo de classificação de veículos (Fase 2 - VMMR):
- Estimativa da cor da lataria (espaço HSV com correção de iluminação)
- Mapeamento de tipo de veículo (Carro, Moto, Caminhão, Ônibus)
- Classificação Zero-Shot de Marca, Modelo e Carroceria via Google SigLIP
  (com Prompt Ensembling da frota brasileira e cache de embeddings de texto em RAM/disco)
"""

import os
import sys
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Mapeamento de classes do modelo COCO para português
MAPA_TIPOS_VEICULO = {
    2: "Carro",
    3: "Moto",
    5: "Ônibus",
    7: "Caminhão",
}

# Catálogo de Prompt Ensembling da frota brasileira (Marca/Modelo -> Categoria + Prompts visuais)
# Cada modelo possui múltiplas descrições visuais (frente, traseira, silhueta) cujos vetores
# matemáticos são combinados (média normalizada) uma única vez na inicialização.
CATALOGO_FROTA_BR = {
    "Volkswagen Fox": {
        "categoria": "Hatchback",
        "prompts": [
            "rear view of a Volkswagen Fox hatchback car in Brazil",
            "rear view of a Volkswagen Fox with oval taillights and high roof",
            "rear view of a silver Volkswagen Fox car with circular taillight reflectors and centered VW logo",
            "rear view of a Volkswagen Fox tall boy hatchback",
            "front view of a Volkswagen Fox compact hatchback car showing front grille and headlights",
            "front angle of a Volkswagen Fox car on a street",
        ],
    },
    "Volkswagen Gol": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Volkswagen Gol subcompact hatchback car with low front hood, horizontal grille and VW badge",
            "front angle of a standard low-roof Volkswagen Gol hatchback passenger car in Brazil",
            "rear view of a Volkswagen Gol compact hatchback car with low roofline and squarish horizontal taillights flanking the tailgate",
            "rear angle of a Volkswagen Gol economy hatchback showing compact rear bumper and VW logo",
            "a photo of a Volkswagen Gol subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Volkswagen Polo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a modern Volkswagen Polo compact hatchback car with sharp angular headlights, slim front grille and VW emblem",
            "front angle of a modern Volkswagen Polo hatchback showing sculpted hood creases and sporty front bumper",
            "rear view of a Volkswagen Polo hatchback car with sharp geometric wraparound taillights",
            "rear angle of a modern Volkswagen Polo compact hatchback car in Brazil",
            "a photo of a modern Volkswagen Polo subcompact two-box hatchback car on a road",
        ],
    },
    "Volkswagen Up": {
        "categoria": "Hatchback",
        "prompts": [
            "front view of a tiny Volkswagen Up micro city car with short snub nose hood and rounded headlights",
            "front three-quarters view of an ultra-compact Volkswagen Up micro city car with short front overhang",
            "rear view of a tiny Volkswagen Up subcompact city car with vertical corner taillights and flat vertical rear hatch",
            "rear angle of a tiny Volkswagen Up micro city car showing flat vertical tailgate without roof rails",
            "a photo of an ultra-compact Volkswagen Up micro city car in Brazil",
        ],
    },
    "Volkswagen Golf": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Volkswagen Golf medium hatchback car with wide horizontal grille, sleek headlights and iconic VW emblem",
            "front angle of a Volkswagen Golf hatchback showing low wide stance, sculpted hood and aggressive lower air dam",
            "rear view of a Volkswagen Golf medium hatchback car with horizontal split taillights and signature wide C-pillar",
            "rear angle of a Volkswagen Golf hatchback showing dual exhaust tips and aerodynamic rear spoiler",
            "a photo of a Volkswagen Golf five-door medium hatchback car on a road",
        ],
    },
    "Chevrolet Onix": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Onix compact hatchback car showing dual-port front grille, headlights and Chevrolet bowtie",
            "front angle of a modern Chevrolet Onix hatchback showing sculpted hood lines and lower air intake",
            "rear view of a Chevrolet Onix compact hatchback car with wide horizontal wraparound taillights",
            "rear angle of a modern Chevrolet Onix hatchback car in Brazil with aerodynamic curved tailgate",
            "a photo of a Chevrolet Onix subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Chevrolet Celta": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a small Chevrolet Celta economy hatchback car showing simple headlights and small front grille with gold bowtie",
            "front angle of a low Chevrolet Celta subcompact hatchback passenger car on a Brazilian street",
            "rear view of a Chevrolet Celta small economy hatchback with high-mounted triangular corner taillights",
            "rear angle of a Chevrolet Celta compact hatchback showing curved rear window and gold Chevrolet emblem",
            "a photo of a small low Chevrolet Celta two-box economy hatchback car in Brazil",
        ],
    },
    "Chevrolet Corsa": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Corsa compact hatchback car showing almond headlights and front grille with Chevrolet badge",
            "front angle of a classic Chevrolet Corsa subcompact hatchback car on a Brazilian street",
            "rear view of a Chevrolet Corsa hatchback with vertical taillights on the rear pillars",
            "rear angle of a compact Chevrolet Corsa hatchback passenger car",
            "a photo of a classic Chevrolet Corsa two-box compact hatchback car in Brazil",
        ],
    },
    "Chevrolet Cruze Sport6": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Cruze Sport6 medium hatchback showing aggressive dual-port grille, headlights and gold bowtie",
            "front angle of a Chevrolet Cruze Sport6 hatchback with sporty front bumper and sculpted hood",
            "rear view of a Chevrolet Cruze Sport6 hatchback car with prominent roof spoiler and wide horizontal taillights",
            "rear angle of a Chevrolet Cruze Sport6 medium hatchback showing aerodynamic fastback roofline",
            "a photo of a modern Chevrolet Cruze Sport6 five-door hatchback car on a Brazilian highway",
        ],
    },
    "Fiat Palio": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Palio compact hatchback car showing rounded headlights, front grille and Fiat emblem",
            "front angle of a Fiat Palio subcompact hatchback car on a Brazilian street",
            "rear view of a Fiat Palio compact hatchback with curved taillights mounted on the rear pillars",
            "rear angle of a Fiat Palio hatchback showing rounded tailgate and Fiat badge",
            "a photo of a Fiat Palio compact two-box hatchback car in Brazil",
        ],
    },
    "Fiat Uno Mille": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic boxy 1980s-2013 Fiat Uno Mille hatchback car with sharp straight angular lines, flat hood, rectangular headlights and narrow grille",
            "front angle of an old boxy Fiat Uno Mille caixote economy hatchback car on a Brazilian street",
            "rear view of a classic boxy Fiat Uno Mille hatchback with straight flat vertical rear glass, flat tailgate and small low rectangular taillights",
            "rear angle of a classic square Fiat Uno Mille economy hatchback showing sharp boxy edges and thin rear bumper",
            "a photo of a classic boxy Fiat Uno Mille two-box economy hatchback car in Brazil",
        ],
    },
    "Fiat Uno (Novo)": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a 2010+ Fiat Novo Uno Vivace Way Attractive compact hatchback showing rounded front fascia with three distinctive rounded-square slots on the grille and large headlights",
            "front angle of a modern rounded-square Fiat Novo Uno hatchback showing three characteristic small square front grille cutouts and tall roofline",
            "rear view of a 2010+ Fiat Novo Uno compact hatchback car with tall vertical taillights mounted high along the C-pillars",
            "rear angle of a Fiat Novo Uno Vivace Way hatchback showing rounded-boxy silhouette, tall C-pillar taillights and compact tailgate",
            "a photo of a 2010+ Fiat Novo Uno Vivace Way compact two-box hatchback car on a Brazilian road",
        ],
    },
    "Fiat Argo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a modern Fiat Argo compact hatchback car showing wide textured grille, sharp headlights and Fiat lettering",
            "front angle of a Fiat Argo hatchback showing sculpted hood and sporty front bumper intake",
            "rear view of a Fiat Argo compact hatchback car with sharp horizontal split LED taillights",
            "rear angle of a modern Fiat Argo hatchback showing integrated tailgate spoiler and rear bumper diffuser",
            "a photo of a modern Fiat Argo subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Fiat Mobi": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny Fiat Mobi subcompact city car showing high front hood, oversized headlights and wide Fiat grille",
            "front angle of a compact Fiat Mobi hatchback on a Brazilian street",
            "rear view of a Fiat Mobi subcompact car featuring distinctive glossy all-black glass rear hatch and corner taillights",
            "rear angle of a tiny Fiat Mobi city car showing vertical flat glass tailgate and compact bumper",
            "a photo of a small Fiat Mobi subcompact urban hatchback car in Brazil",
        ],
    },
    "Fiat Punto": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Punto hatchback car showing aerodynamic sloping front nose, narrow grille and headlights",
            "front angle of a Fiat Punto compact hatchback car with curved hood and sporty front fascia",
            "rear view of a Fiat Punto hatchback car with distinctive tall vertical taillights along the rear pillars",
            "rear angle of a Fiat Punto hatchback showing curved muscular rear shoulders and Fiat badge",
            "a photo of a Fiat Grande Punto compact hatchback car on a Brazilian road",
        ],
    },
    "Hyundai HB20": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Hyundai HB20 compact hatchback car showing large cascading front grille, sharp headlights and Hyundai logo",
            "front angle of a modern Hyundai HB20 hatchback showing sculpted hood creases and geometric fog light surrounds",
            "rear view of a Hyundai HB20 compact hatchback car with sharp angular boomerang taillights or full-width light bar",
            "rear angle of a modern Hyundai HB20 hatchback car in Brazil showing sculpted tailgate and rear bumper",
            "a photo of a Hyundai HB20 subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Ford Ka (Antigo)": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic Brazilian two-door Ford Ka G1 G2 subcompact hatchback car with tiny rounded body, oval or angular headlights and Ford blue oval",
            "front angle of an older two-door Ford Ka G1 G2 micro hatchback car on a Brazilian street",
            "rear view of a classic two-door Ford Ka G1 G2 subcompact car with very short rounded rear end or large vertical pillar taillights",
            "rear angle of an old two-door Ford Ka G1 G2 micro city hatchback showing curved compact tailgate and black or body-colored bumper",
            "a photo of an older two-door Brazilian Ford Ka G1 G2 subcompact hatchback car on a street",
        ],
    },
    "Ford Ka": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a 2014+ Ford Ka G3 four-door compact hatchback car showing large Aston Martin style hexagonal grille, swept headlights and Ford blue oval",
            "front angle of a modern four-door Ford Ka G3 hatchback showing sculpted hood and prominent hexagonal front grille",
            "rear view of a modern four-door Ford Ka G3 compact hatchback car with triangular wraparound corner taillights",
            "rear angle of a 2014+ Ford Ka G3 four-door hatchback showing clean modern tailgate with centered Ford emblem",
            "a photo of a 2014+ Ford Ka G3 four-door compact two-box hatchback passenger car on a Brazilian road",
        ],
    },
    "Ford Fiesta Rocam": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a 2003-2014 Brazilian Ford Fiesta Rocam Mk5 Mk6 hatchback car with large rounded triangular headlights, small narrow front grille and Ford oval badge",
            "front angle of a Brazilian Ford Fiesta Rocam hatchback car showing compact rounded hood and simple lower bumper",
            "rear view of a 2003-2014 Ford Fiesta Rocam compact hatchback with tall upright rear end and vertical taillights mounted high on the C-pillars",
            "rear angle of a Brazilian Ford Fiesta Rocam hatchback showing tall C-pillar vertical rear lamps and upright tailgate",
            "a photo of a 2003-2014 Brazilian Ford Fiesta Rocam five-door compact hatchback car on a street",
        ],
    },
    "Ford New Fiesta": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a 2011-2019 Ford New Fiesta global hatchback car with large Aston Martin style trapezoidal front grille, sharp swept-back slim headlights and Ford badge",
            "front angle of a sporty Ford New Fiesta hatchback showing sculpted hood creases and aggressive trapezoidal grille",
            "rear view of a Ford New Fiesta hatchback car with wide horizontal split wraparound taillights and sculpted tailgate",
            "rear angle of a 2011-2019 Ford New Fiesta global hatchback showing rear roof spoiler, sloping roofline and horizontal taillights",
            "a photo of a modern 2011-2019 Ford New Fiesta global five-door hatchback car on a Brazilian road",
        ],
    },
    "Renault Kwid": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Renault Kwid compact mini crossover hatchback showing high front hood, black plastic wheel arches and large Renault chrome diamond",
            "front angle of a Renault Kwid subcompact hatchback showing split daytime running lights and rugged front bumper",
            "rear view of a small Renault Kwid compact car with rounded rectangular taillights and tall rear bumper",
            "rear angle of a Renault Kwid mini crossover hatchback showing elevated ride height and compact tailgate",
            "a photo of an entry-level Renault Kwid subcompact crossover hatchback car on a Brazilian street",
        ],
    },
    "Renault Sandero": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Renault Sandero compact hatchback car showing wide front grille, headlights and large Renault chrome diamond",
            "front angle of a Renault Sandero hatchback showing wide stance, contoured hood and lower fog lamps",
            "rear view of a Renault Sandero compact hatchback car with horizontal taillights extending into the tailgate",
            "rear angle of a spacious Renault Sandero hatchback car in Brazil showing wide tailgate and Renault emblem",
            "a photo of a Renault Sandero subcompact two-box hatchback passenger car on a Brazilian road",
        ],
    },
    "Peugeot 208": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Peugeot 208 compact hatchback car showing dramatic saber-tooth LED daytime running lights, wide front grille and Peugeot lion",
            "front angle of a modern Peugeot 208 hatchback showing sculpted hood and aggressive front bumper intake",
            "rear view of a Peugeot 208 hatchback car with glossy black horizontal band connecting three-claw LED taillights",
            "rear angle of a modern Peugeot 208 compact hatchback showing sporty rear spoiler and curved tailgate",
            "a photo of a modern Peugeot 208 subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Citroën C3": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Citroen C3 compact hatchback car showing split headlights, double chevron chrome grille and high rounded hood",
            "front angle of a Citroen C3 hatchback showing elevated stance and black lower bumper claddings",
            "rear view of a Citroen C3 compact hatchback car with rectangular C-shaped taillights and upright tailgate",
            "rear angle of a modern Citroen C3 hatchback showing rounded rear bumper and double chevron badge",
            "a photo of a Citroen C3 compact five-door hatchback car on a Brazilian road",
        ],
    },
    "Honda Fit": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Honda Fit Jazz compact tall hatchback showing steeply sloped windshield, short hood and Honda H grille",
            "front angle of a Honda Fit MPV-style hatchback showing swept-back headlights and aerodynamic front bumper",
            "rear view of a Honda Fit Jazz hatchback car with tall vertical taillights flanking the rear glass window",
            "rear angle of a Honda Fit tall compact hatchback showing large vertical tailgate and chrome rear trim",
            "a photo of a Honda Fit subcompact one-box hatchback car on a Brazilian road",
        ],
    },
    "Toyota Etios": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Toyota Etios compact hatchback car showing smiling front grille, headlights and Toyota emblem",
            "front angle of a Toyota Etios subcompact hatchback car on a Brazilian street",
            "rear view of a Toyota Etios compact hatchback car with simple vertical corner taillights and flat tailgate",
            "rear angle of a Toyota Etios hatchback showing compact rear bumper and centered Toyota badge",
            "a photo of a Toyota Etios subcompact two-box hatchback passenger car in Brazil",
        ],
    },
    "Toyota Yaris": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Toyota Yaris compact hatchback car showing large black lower grille, slim headlights and Toyota badge",
            "front angle of a Toyota Yaris hatchback showing sculpted front bumper, hood lines and integrated fog lights",
            "rear view of a Toyota Yaris compact hatchback car with horizontal wraparound split taillights",
            "rear angle of a modern Toyota Yaris hatchback showing sporty rear diffuser and tailgate spoiler",
            "a photo of a Toyota Yaris subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "BYD Dolphin": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a BYD Dolphin electric compact hatchback car showing closed aerodynamic grille, modern LED headlights and BYD badge",
            "front angle of a BYD Dolphin EV hatchback showing short electric vehicle hood and two-tone exterior",
            "rear view of a BYD Dolphin EV hatchback car featuring full-width continuous LED light bar with geometric knot pattern",
            "rear angle of a modern BYD Dolphin electric hatchback showing futuristic rear spoiler and smooth bumper",
            "a photo of a BYD Dolphin all-electric compact hatchback car on a Brazilian road",
        ],
    },
    "Toyota Corolla": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Toyota Corolla medium sedan car showing wide lower trapezoidal grille, sharp LED headlights and Toyota emblem",
            "front angle of a modern Toyota Corolla sedan showing sculpted hood and aerodynamic three-box sedan silhouette",
            "rear view of a Toyota Corolla four-door sedan car with elongated separate rear trunk and horizontal LED taillights",
            "rear angle of a Toyota Corolla medium sedan showing separate boot lid, chrome accent bar and rear bumper",
            "a photo of a Toyota Corolla four-door three-box sedan passenger car on a Brazilian road",
        ],
    },
    "Honda Civic": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Honda Civic medium sedan car showing sporty solid wing front grille, jewel-eye headlights and Honda emblem",
            "front angle of a low aggressive Honda Civic sedan showing sculpted hood creases and wide stance",
            "rear view of a Honda Civic four-door sedan car with elongated separate trunk and distinctive boomerang C-shaped LED taillights",
            "rear angle of a sporty Honda Civic three-box sedan showing fastback trunk lid and dual exhaust cutouts",
            "a photo of a Honda Civic four-door passenger sedan car on a Brazilian road",
        ],
    },
    "Honda City": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Honda City compact sedan car showing thick chrome front grille bar, sharp headlights and Honda emblem",
            "front angle of a modern Honda City sedan showing aerodynamic front nose and sculpted three-box sedan proportions",
            "rear view of a Honda City four-door compact sedan car with elongated separate rear trunk and wraparound LED taillights",
            "rear angle of a Honda City sedan showing distinct rear boot lid, integrated lip spoiler and Honda badge",
            "a photo of a Honda City four-door subcompact three-box sedan car on a Brazilian road",
        ],
    },
    "Chevrolet Onix Plus": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Onix Plus compact sedan car showing dual-port grille, headlights and Chevrolet gold bowtie",
            "front angle of a modern Chevrolet Onix Plus sedan showing sleek hood lines and three-box sedan profile",
            "rear view of a Chevrolet Onix Plus four-door sedan car with elongated separate rear trunk and horizontal split taillights",
            "rear angle of a Chevrolet Onix Plus compact sedan showing separate rear trunk lid and Chevrolet badge",
            "a photo of a Chevrolet Onix Plus subcompact three-box sedan passenger car on a Brazilian road",
        ],
    },
    "Chevrolet Prisma (G1)": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a 2006-2012 first generation Chevrolet Prisma G1 compact sedan derived from Celta with straight simple front fascia, triangular headlights and gold bowtie",
            "front angle of a 2006-2012 Chevrolet Prisma G1 economy sedan showing Celta-style hood and simple front grille",
            "rear view of a first generation Chevrolet Prisma G1 four-door sedan with short elevated high rear trunk and triangular corner taillights",
            "rear angle of a 2006-2012 Chevrolet Prisma G1 three-box economy sedan showing tall short boot lid and gold Chevrolet emblem",
            "a photo of a 2006-2012 first generation Chevrolet Prisma G1 compact four-door sedan car in Brazil",
        ],
    },
    "Chevrolet Prisma (G2)": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a 2013-2019 second generation Chevrolet Prisma G2 Joy Plus sedan derived from Onix with Chevrolet dual-port grille, large headlights and gold bowtie",
            "front angle of a Chevrolet Prisma G2 Joy Plus four-door sedan showing rounded sculpted hood and dual-port front fascia",
            "rear view of a 2013-2019 Chevrolet Prisma G2 sedan car with fluid elongated separate rear trunk and horizontal wraparound taillights",
            "rear angle of a Chevrolet Prisma G2 Joy Plus three-box sedan showing smooth aerodynamic trunk lid, high C-pillar and Chevrolet emblem",
            "a photo of a 2013-2019 Chevrolet Prisma G2 Joy Plus four-door compact three-box sedan car on a Brazilian road",
        ],
    },
    "Chevrolet Cruze": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Cruze medium sedan car showing dual-port aerodynamic grille, projector headlights and bowtie",
            "front angle of an elegant Chevrolet Cruze four-door sedan showing sculpted hood and chrome fog lamp bezels",
            "rear view of a Chevrolet Cruze medium sedan car with elongated separate rear trunk lid and horizontal taillights",
            "rear angle of a Chevrolet Cruze three-box sedan showing wide rear bumper, trunk lip and Chevrolet badge",
            "a photo of a Chevrolet Cruze four-door medium passenger sedan car on a Brazilian highway",
        ],
    },
    "Chevrolet Classic": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a small Chevrolet Classic compact sedan car showing simple front grille, trapezoidal headlights and gold bowtie",
            "front angle of a classic Chevrolet Corsa Classic four-door economy sedan on a Brazilian street",
            "rear view of a Chevrolet Classic small sedan car with tall separate rear trunk box and corner taillights",
            "rear angle of a Chevrolet Classic compact three-box sedan showing distinctive trunk lid with gold bowtie",
            "a photo of an economy Chevrolet Classic four-door three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Voyage": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Voyage compact sedan car showing horizontal front grille, headlights and VW emblem",
            "front angle of a Volkswagen Voyage four-door subcompact sedan car on a Brazilian road",
            "rear view of a Volkswagen Voyage compact sedan car with elongated separate rear trunk and horizontal taillights",
            "rear angle of a Volkswagen Voyage three-box sedan showing distinct boot lid with centered VW logo",
            "a photo of a Volkswagen Voyage four-door subcompact three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Virtus": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Virtus compact sedan car showing wide chrome grille, sharp angular headlights and VW logo",
            "front angle of a modern Volkswagen Virtus four-door sedan showing sculpted hood and lower bumper air intake",
            "rear view of a Volkswagen Virtus compact sedan car with elongated separate rear trunk and slim horizontal LED taillights",
            "rear angle of a Volkswagen Virtus three-box sedan showing elegant separate trunk lid and chrome rear trim",
            "a photo of a modern Volkswagen Virtus four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Volkswagen Jetta": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Jetta medium executive sedan car showing wide hexagonal chrome grille, LED headlights and VW emblem",
            "front angle of a Volkswagen Jetta four-door sedan showing long hood creases and muscular front fascia",
            "rear view of a Volkswagen Jetta medium sedan car with elongated separate rear trunk lid and sharp horizontal LED taillights",
            "rear angle of an executive Volkswagen Jetta three-box sedan showing dual exhaust surrounds and wide trunk",
            "a photo of a Volkswagen Jetta four-door executive three-box sedan car on a Brazilian highway",
        ],
    },
    "Volkswagen Santana": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a classic Volkswagen Santana three-box sedan car showing rectangular headlights, horizontal front grille and VW emblem",
            "front angle of a classic boxy Volkswagen Santana four-door sedan car on a Brazilian street",
            "rear view of a classic Volkswagen Santana sedan car with flat boxy separate rear trunk and wide rectangular horizontal taillights",
            "rear angle of a Volkswagen Santana four-door three-box sedan showing long flat trunk lid, squared roofline and chrome trim",
            "a photo of a classic Volkswagen Santana four-door three-box passenger sedan car in Brazil",
        ],
    },
    "Fiat Siena": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a classic Fiat Siena Fire EL compact sedan car based on Palio with narrow front fascia, rounded headlights and small Fiat grille",
            "front angle of an older classic Fiat Siena four-door subcompact economy sedan on a Brazilian street",
            "rear view of a classic Fiat Siena Fire EL sedan car with compact separate trunk and smaller rectangular horizontal taillights on the trunk lid",
            "rear angle of a classic Fiat Siena three-box economy sedan showing flat narrow rear end and Fiat emblem",
            "a photo of a classic Fiat Siena Fire EL four-door compact three-box sedan car in Brazil",
        ],
    },
    "Fiat Grand Siena": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Fiat Grand Siena compact sedan car with larger taller body, wide front grille with horizontal chrome bar, swept-back headlights and Fiat badge",
            "front angle of a Fiat Grand Siena four-door sedan showing tall wide hood, chrome grille trim and lower bumper fog lamps",
            "rear view of a Fiat Grand Siena sedan car with large tall separate rear trunk and large wraparound taillights extending onto the trunk lid",
            "rear angle of a Fiat Grand Siena three-box sedan showing taller muscular rear quarters, elevated boot lid and large wraparound rear lamps",
            "a photo of a Fiat Grand Siena four-door compact three-box sedan passenger car on a Brazilian road",
        ],
    },
    "Fiat Cronos": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Fiat Cronos compact sedan car showing stylish textured grille, sculpted hood, headlights and Fiat logo",
            "front angle of a modern Fiat Cronos four-door sedan showing aggressive lower air intake and LED daytime lights",
            "rear view of a Fiat Cronos compact sedan car with elongated separate rear trunk and slim horizontal LED taillights",
            "rear angle of a Fiat Cronos three-box sedan showing sculpted rear trunk lid with integrated ducktail spoiler",
            "a photo of a modern Fiat Cronos four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Hyundai HB20S": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Hyundai HB20S compact sedan car showing large cascading front grille, sharp headlights and Hyundai logo",
            "front angle of a modern Hyundai HB20S four-door sedan showing sculpted hood creases and geometric bumper styling",
            "rear view of a Hyundai HB20S compact sedan car with elongated separate rear trunk and sharp angular taillights",
            "rear angle of a Hyundai HB20S three-box sedan showing fastback-style separate trunk lid and rear bumper",
            "a photo of a Hyundai HB20S four-door subcompact three-box sedan car on a Brazilian road",
        ],
    },
    "Renault Logan": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Renault Logan compact sedan car showing wide front grille, headlights and large Renault chrome diamond",
            "front angle of a Renault Logan four-door sedan showing tall front hood and upright three-box sedan profile",
            "rear view of a Renault Logan compact sedan car with tall boxy separate rear trunk and vertical rectangular taillights",
            "rear angle of a spacious Renault Logan three-box sedan showing wide flat trunk lid and Renault badge",
            "a photo of a Renault Logan four-door boxy compact sedan car on a Brazilian street",
        ],
    },
    "Nissan Versa": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Nissan Versa compact sedan car showing bold chrome V-motion front grille, sharp headlights and Nissan logo",
            "front angle of a modern Nissan Versa four-door sedan showing sculpted hood and lower front air dam",
            "rear view of a Nissan Versa compact sedan car with elongated separate rear trunk and wide boomerang split taillights",
            "rear angle of a Nissan Versa three-box sedan showing floating roof C-pillar and separate rear boot lid",
            "a photo of a modern Nissan Versa four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Nissan Sentra": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Nissan Sentra medium sedan car showing wide bold V-motion chrome grille, slim LED headlights and Nissan logo",
            "front angle of an aggressive Nissan Sentra four-door sedan showing low sculpted hood and wide sports sedan stance",
            "rear view of a Nissan Sentra medium sedan car with elongated separate rear trunk and sharp horizontal LED taillights",
            "rear angle of a Nissan Sentra executive three-box sedan showing sculpted trunk lid with integrated lip spoiler",
            "a photo of a modern Nissan Sentra four-door medium passenger sedan car on a Brazilian highway",
        ],
    },
    "Jeep Renegade": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Jeep Renegade compact boxy SUV with roof rack rails on top, tall ground clearance, iconic seven-slot grille and round headlights",
            "front angle of a Jeep Renegade compact crossover SUV showing high upright hood, black bumper cladding and round projector lamps",
            "rear view of a boxy Jeep Renegade compact SUV with roof rails and distinctive square taillights with embossed X pattern",
            "rear angle of a Jeep Renegade SUV showing tall upright tailgate, roof rack rails, black fender flares and high ground clearance",
            "a photo of a boxy Jeep Renegade compact crossover SUV car on a Brazilian road",
        ],
    },
    "Jeep Compass": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Jeep Compass medium SUV with roof rack rails on top, tall ground clearance, seven-slot front grille and slim LED headlights",
            "front angle of a modern Jeep Compass crossover SUV showing high sculpted hood, black wheel arch moldings and Jeep emblem",
            "rear view of a Jeep Compass medium SUV with roof rack rails on top and slim horizontal wraparound LED taillights",
            "rear angle of a Jeep Compass SUV showing dual-tone floating roof with rails, elevated suspension and wide tailgate",
            "a photo of a modern Jeep Compass medium crossover SUV car on a Brazilian road",
        ],
    },
    "Volkswagen T-Cross": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen T-Cross compact crossover SUV with roof rack rails on top, wide horizontal grille and prominent VW emblem",
            "a photo of a Volkswagen T-Cross subcompact SUV crossover car in Brazil",
            "front angle of a modern Volkswagen T-Cross SUV showing tall front hood, roof rails and headlights",
            "a photo of a Volkswagen T-Cross SUV in Brazil with roof rack rails on top and wide horizontal grille",
            "a photo of a dark Volkswagen T-Cross compact crossover SUV car in Brazil",
            "rear view of a Volkswagen T-Cross compact SUV with black horizontal reflective light bar across the tailgate",
        ],
    },
    "Volkswagen Nivus": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen Nivus coupe crossover SUV with roof rack rails on top, piano black grille, LED headlights and VW emblem",
            "front angle of a modern Volkswagen Nivus compact fastback crossover showing sculpted hood creases, tall ground clearance and fog lights",
            "rear view of a Volkswagen Nivus coupe SUV with sloping fastback rear window and gloss black horizontal band connecting taillights",
            "rear angle of a Volkswagen Nivus Taigo coupe crossover SUV showing sleek sloping roofline with roof rails and rear spoiler",
            "a photo of a modern Volkswagen Nivus compact coupe SUV crossover car on a Brazilian road",
        ],
    },
    "Hyundai Creta": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Hyundai Creta compact SUV with roof rack rails on top, tall ground clearance, large parametric front grille and split LED headlights",
            "front angle of a modern Hyundai Creta crossover SUV showing high commanding hood, black wheel arch cladding and Hyundai emblem",
            "rear view of a Hyundai Creta compact SUV with roof rack rails on top and distinctive geometric split LED taillights",
            "rear angle of a Hyundai Creta crossover SUV showing silver roof arch accent, roof rails and sculpted tailgate",
            "a photo of a modern Hyundai Creta compact crossover SUV car on a Brazilian road",
        ],
    },
    "Hyundai Tucson": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Hyundai Tucson medium SUV with roof rack rails on top, tall ground clearance, bold front grille and LED headlights",
            "front angle of a modern Hyundai Tucson crossover SUV showing sculpted muscular front fenders and high ground clearance",
            "rear view of a Hyundai Tucson medium SUV with roof rack rails on top and horizontal LED taillights",
            "rear angle of a Hyundai Tucson crossover SUV showing elevated suspension, roof rails and wide rear tailgate",
            "a photo of a Hyundai Tucson medium crossover SUV car on a Brazilian road",
        ],
    },
    "Chevrolet Tracker": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Chevrolet Tracker compact SUV with roof rack rails on top, tall ground clearance, dual-port grille and Chevrolet bowtie",
            "front angle of a modern Chevrolet Tracker crossover SUV showing high front hood, black lower bumper cladding and slim headlights",
            "rear view of a Chevrolet Tracker compact SUV with roof rack rails on top and high-mounted horizontal wraparound taillights",
            "rear angle of a Chevrolet Tracker crossover SUV showing elevated ground clearance, black wheel arches and roof rails",
            "a photo of a modern Chevrolet Tracker subcompact crossover SUV car on a Brazilian road",
        ],
    },
    "Nissan Kicks": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Nissan Kicks compact crossover SUV with roof rack rails on top, large V-motion front grille, sharp headlights and Nissan logo",
            "front angle of a modern Nissan Kicks SUV showing tall front hood, black lower bumper claddings and elevated ground clearance",
            "rear view of a Nissan Kicks compact SUV with roof rack rails on top and boomerang connected LED taillights",
            "rear angle of a Nissan Kicks crossover SUV showing floating roof design with roof rails and sporty tailgate spoiler",
            "a photo of a modern Nissan Kicks compact crossover SUV car on a Brazilian road",
        ],
    },
    "Honda HR-V": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Honda HR-V compact crossover SUV with roof rack rails on top, tall ground clearance, horizontal slat grille and slim LED headlights",
            "front angle of a modern Honda HR-V SUV showing sleek front hood, black wheel arch cladding and Honda emblem",
            "rear view of a Honda HR-V compact SUV with roof rack rails on top and full-width continuous horizontal LED taillight bar",
            "rear angle of a Honda HR-V coupe SUV showing hidden rear door handles in C-pillar, roof rails and sloping tailgate",
            "a photo of a modern Honda HR-V compact crossover SUV car on a Brazilian road",
        ],
    },
    "Renault Duster": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Renault Duster rugged compact SUV with heavy roof rack rails on top, tall ground clearance, muscular front bumper and Renault diamond",
            "front angle of a rugged Renault Duster SUV showing wide flared wheel arches, front skid plate and elevated stance",
            "rear view of a Renault Duster compact SUV with roof rack rails on top and distinctive square taillights with cross pattern",
            "rear angle of a Renault Duster rugged crossover SUV showing prominent roof rails, high ground clearance and wide tailgate",
            "a photo of a Renault Duster rugged compact SUV car on a Brazilian road",
        ],
    },
    "Ford EcoSport": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Ford EcoSport compact SUV with roof rack rails on top, tall ground clearance, large trapezoidal grille and Ford blue oval",
            "front angle of a Ford EcoSport mini SUV showing high hood, lower bumper cladding and projector headlights",
            "rear view of a Ford EcoSport compact SUV with distinctive external spare tire mounted on the rear swing tailgate and roof rack rails",
            "rear angle of a Ford EcoSport mini crossover SUV showing high ground clearance, roof rails and tailgate-mounted spare wheel",
            "a photo of a Ford EcoSport compact crossover SUV car on a Brazilian road",
        ],
    },
    "Fiat Pulse": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Fiat Pulse compact crossover SUV with roof rack rails on top, tall ground clearance, wide front grille and Fiat script logo",
            "front angle of a modern Fiat Pulse SUV showing sculpted high hood, black wheel arch cladding and LED headlights",
            "rear view of a Fiat Pulse compact crossover SUV with roof rack rails on top and horizontal tapered LED taillights",
            "rear angle of a Fiat Pulse compact SUV showing elevated suspension, roof rails and sporty rear bumper",
            "a photo of a modern Fiat Pulse compact crossover SUV car on a Brazilian road",
        ],
    },
    "Fiat Fastback": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Fiat Fastback coupe crossover SUV with roof rack rails on top, tall ground clearance, honeycomb grille and Fiat logo",
            "front angle of a Fiat Fastback SUV showing muscular front bumper, fog lamp intakes and full-LED headlights",
            "rear view of a Fiat Fastback coupe SUV with elevated fastback tail, integrated ducktail spoiler and slim horizontal LED taillights",
            "rear angle of a Fiat Fastback coupe crossover showing sweeping coupe roofline with roof rails and tall ground clearance",
            "a photo of a modern Fiat Fastback coupe crossover SUV car on a Brazilian road",
        ],
    },
    "Toyota Corolla Cross": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Toyota Corolla Cross medium SUV with roof rack rails on top, tall ground clearance, double trapezoid front grille and Toyota emblem",
            "front angle of a Toyota Corolla Cross crossover SUV showing high sculpted hood, black fender cladding and sleek LED headlights",
            "rear view of a Toyota Corolla Cross medium SUV with roof rack rails on top and elongated horizontal LED taillights",
            "rear angle of a Toyota Corolla Cross SUV showing elevated suspension, roof rails on top and wide rear tailgate",
            "a photo of a modern Toyota Corolla Cross medium crossover SUV car on a Brazilian road",
        ],
    },
    "Toyota SW4": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large Toyota SW4 Fortuner SUV with roof rack rails on top, massive ground clearance, imposing chrome grille and LED headlights",
            "front angle of a large Toyota SW4 SUV showing high commanding front hood, fog lamp bezels and prominent Toyota emblem",
            "rear view of a large Toyota SW4 Fortuner SUV with roof rack rails on top, tall rear bumper and horizontal LED taillights with chrome SW4 bar",
            "rear angle of a heavy Toyota SW4 full-size SUV showing high ground clearance, side running boards, roof rails and rear spoiler",
            "a photo of a large Toyota SW4 Fortuner four-wheel drive SUV on a Brazilian road",
        ],
    },
    "Fiat Strada": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Fiat Strada compact pickup truck showing bold front grille with Fiat script logo, Italian flag badge and headlights",
            "front angle of a modern Fiat Strada small utility pickup truck on a Brazilian road",
            "rear view of a Fiat Strada compact pickup truck showing open rear cargo bed, vertical taillights and tailgate with Fiat lettering",
            "rear angle of a Fiat Strada double-cab pickup truck showing open cargo bed with bedliner, sports bar and utility stance",
            "a photo of a Fiat Strada compact utility pickup truck car in Brazil",
        ],
    },
    "Volkswagen Saveiro": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Volkswagen Saveiro compact utility pickup truck showing front grille, headlights and VW emblem",
            "front angle of a Volkswagen Saveiro small pickup truck with lower front bumper and fog lights",
            "rear view of a Volkswagen Saveiro compact pickup truck showing open rear cargo bed and tailgate with VW logo",
            "rear angle of a Volkswagen Saveiro utility pickup truck showing open cargo bed, sports bar roll cage and low side steps",
            "a photo of a Volkswagen Saveiro subcompact utility pickup truck on a Brazilian road",
        ],
    },
    "Fiat Toro": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Fiat Toro medium pickup truck showing split two-tier LED headlights, bold front grille and large Fiat lettering",
            "front angle of a modern Fiat Toro pickup truck showing muscular sculpted hood and rugged front skid plate",
            "rear view of a Fiat Toro medium pickup truck featuring unique dual-opening barn-door split tailgate and vertical taillights",
            "rear angle of a Fiat Toro pickup truck showing open rear cargo bed, sports bar roll bar, roof rails and utility stance",
            "a photo of a Fiat Toro compact unibody pickup truck on a Brazilian road",
        ],
    },
    "Toyota Hilux": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Toyota Hilux double-cab pickup truck showing massive trapezoidal front grille, headlights and Toyota emblem",
            "front angle of a rugged Toyota Hilux 4x4 pickup truck showing high front bumper, skid plate and commanding stance",
            "rear view of a Toyota Hilux pickup truck showing open rear cargo bed, vertical corner taillights and tailgate handle",
            "rear angle of a Toyota Hilux double-cab pickup truck showing large open cargo bed, chrome sports bar roll bar and side steps",
            "a photo of a rugged Toyota Hilux double-cab four-wheel drive pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet S10": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Chevrolet S10 pickup truck showing wide horizontal chrome grille bar, headlights and Chevrolet bowtie",
            "front angle of a Chevrolet S10 double-cab pickup truck showing high commanding hood and lower bumper fog lamps",
            "rear view of a Chevrolet S10 pickup truck showing open rear cargo bed and tailgate with Chevrolet emblem",
            "rear angle of a Chevrolet S10 double-cab pickup truck showing open cargo bed with bedliner, sports bar roll bar and side steps",
            "a photo of a Chevrolet S10 midsize utility pickup truck on a Brazilian road",
        ],
    },
    "Ford Ranger": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Ford Ranger double-cab pickup truck showing bold front grille, signature C-clamp headlights and Ford emblem",
            "front angle of a modern Ford Ranger 4x4 pickup truck showing high muscular hood and rugged steel skid plate",
            "rear view of a Ford Ranger pickup truck showing open rear cargo bed and tailgate with stamped Ranger lettering",
            "rear angle of a Ford Ranger double-cab pickup truck showing open cargo bed with sports bar roll bar and elevated suspension",
            "a photo of a Ford Ranger double-cab utility pickup truck on a Brazilian road",
        ],
    },
    "Volkswagen Amarok": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Volkswagen Amarok double-cab pickup truck showing wide chrome horizontal grille, headlights and VW logo",
            "front angle of a Volkswagen Amarok 4x4 pickup truck showing muscular squared wheel arches and wide front bumper",
            "rear view of a Volkswagen Amarok pickup truck showing wide open rear cargo bed and tailgate with centered VW emblem",
            "rear angle of a Volkswagen Amarok double-cab pickup truck showing open cargo bed, chrome sports bar roll bar and 4Motion badge",
            "a photo of a Volkswagen Amarok double-cab midsize pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet Montana": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Chevrolet Montana compact pickup truck showing split front headlights, dark grille and Chevrolet bowtie",
            "front angle of a modern Chevrolet Montana pickup truck showing sculpted hood and lower bumper skid plate",
            "rear view of a Chevrolet Montana compact pickup truck showing open rear cargo bed and modern tailgate with black accent trim",
            "rear angle of a Chevrolet Montana double-cab pickup truck showing open cargo bed, bedliner, sports bar and roof rails",
            "a photo of a modern Chevrolet Montana compact utility pickup truck on a Brazilian road",
        ],
    },
    "Renault Oroch": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Renault Oroch double-cab pickup truck showing rugged front bumper, headlights and Renault chrome diamond",
            "front angle of a Renault Duster Oroch pickup truck showing flared muscular fenders and high ground clearance",
            "rear view of a Renault Oroch pickup truck showing open rear cargo bed, tailgate with Oroch lettering and vertical taillights",
            "rear angle of a Renault Oroch double-cab pickup truck showing open cargo bed with black tubular sports bar roll bar",
            "a photo of a Renault Oroch compact double-cab utility pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet Spin": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Chevrolet Spin multi-purpose family minivan MPV showing tall front hood, dual-port grille, headlights and Chevrolet bowtie",
            "front angle of a Chevrolet Spin minivan car on a Brazilian street showing tall roofline and large windshield",
            "rear view of a Chevrolet Spin large family minivan MPV with high roofline, wide vertical rear hatch and horizontal taillights",
            "rear angle of a Chevrolet Spin 7-seater family minivan MPV showing tall boxy passenger cabin and roof rack rails on top",
            "a photo of a Chevrolet Spin family multi-purpose minivan MPV car in Brazil",
        ],
    },
    "Fiat Fiorino": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a white Fiat Fiorino small cargo delivery box van showing compact front cab, headlights and Fiat badge",
            "front angle of a Fiat Fiorino commercial delivery panel van on a Brazilian street",
            "rear view of a white Fiat Fiorino utility cargo van showing tall cubic enclosed cargo box and asymmetric vertical split rear cargo doors",
            "rear angle of a Fiat Fiorino commercial panel van showing tall white delivery cargo box with no rear side windows",
            "a photo of a white Fiat Fiorino small commercial delivery cargo box van in Brazil",
        ],
    },
    # --- Hatches / Subcompactos adicionados ---
    "Peugeot 206": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic Peugeot 206 subcompact hatchback car with distinctive swept-back almond headlights, hood air vent and Peugeot lion emblem",
            "front angle of a Peugeot 206 three-door or five-door hatchback car with low rounded nose and wide lower bumper grille",
            "rear view of a Peugeot 206 compact hatchback car with curved ruby-red corner taillights and rounded tailgate",
            "rear angle of a Peugeot 206 hatchback showing sloping rear hatch glass and black bumper molding",
            "a photo of a Peugeot 206 compact European two-box hatchback car in Brazil",
        ],
    },
    "Peugeot 207": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Brazilian Peugeot 207 hatchback car with large wide smiling black front grille, long sharp headlights and prominent Peugeot lion on the nose",
            "front angle of a Peugeot 207 compact hatchback showing aggressive wide front air intake and fog lights",
            "rear view of a Peugeot 207 compact hatchback car with modernized clear-lens red corner taillights and curved tailgate",
            "rear angle of a Peugeot 207 five-door hatchback car showing compact proportions and rounded rear bumper",
            "a photo of a Peugeot 207 compact two-box hatchback car on a Brazilian street",
        ],
    },
    "Peugeot 307": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Peugeot 307 medium hatchback car with steeply raked windshield, large swept headlights and wide lower grille",
            "front angle of a Peugeot 307 five-door medium hatchback showing long sloping hood and Peugeot lion badge",
            "rear view of a Peugeot 307 medium hatchback car with tall triangular corner taillights and wide flat tailgate",
            "rear angle of a Peugeot 307 medium hatchback showing tall upright C-pillars and spacious cabin",
            "a photo of a Peugeot 307 five-door medium hatchback car in Brazil",
        ],
    },
    "Peugeot 308": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Peugeot 308 medium hatchback car with sculpted hood, chrome-framed grille, LED headlights and Peugeot lion emblem",
            "front angle of a modern Peugeot 308 five-door hatchback showing sporty lower bumper and fog lamp bezels",
            "rear view of a Peugeot 308 medium hatchback car with wide wraparound boomerang or three-claw taillights",
            "rear angle of a Peugeot 308 medium hatchback showing aerodynamic roof spoiler and sculpted tailgate",
            "a photo of a Peugeot 308 five-door medium hatchback passenger car on a Brazilian road",
        ],
    },
    "Citroën C4": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Citroen C4 medium hatchback car with boomerang headlights, double chevron chrome grille bars and curved hood",
            "front angle of a Citroen C4 five-door or VTR coupe hatchback on a Brazilian street",
            "rear view of a Citroen C4 hatchback car with curved split rear glass or high boomerang C-pillar taillights",
            "rear angle of a Citroen C4 medium hatchback showing rounded aerodynamic tail and double chevron emblem",
            "a photo of a Citroen C4 medium hatchback passenger car in Brazil",
        ],
    },
    "Citroën C3 Aircross": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Citroen C3 Aircross compact crossover SUV with roof rack rails, black plastic fender cladding, high ground clearance and double chevron grille",
            "front angle of a Citroen C3 Aircross rugged MPV crossover showing split headlights, silver skid plate and tall windshield",
            "rear view of a Citroen C3 Aircross crossover SUV with vertical pillar taillights and external spare wheel or upright SUV tailgate",
            "rear angle of a Citroen C3 Aircross tall crossover car showing prominent roof rails and rugged black bumper cladding",
            "a photo of a Citroen C3 Aircross compact crossover SUV car on a Brazilian road",
        ],
    },
    "Ford Focus": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Ford Focus medium hatchback car with trapezoidal or Aston Martin style grille, swept headlights and Ford blue oval",
            "front angle of a sporty Ford Focus five-door hatchback showing sculpted hood creases and wide stance",
            "rear view of a Ford Focus medium hatchback car with horizontal wraparound taillights or tall C-pillar lamps and roof spoiler",
            "rear angle of a Ford Focus five-door hatchback showing dynamic rear quarter lines and Ford badge",
            "a photo of a Ford Focus medium five-door hatchback car on a Brazilian road",
        ],
    },
    "Chevrolet Astra": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Astra Opel Astra G medium hatchback car with angular trapezoidal headlights, slim horizontal grille and centered emblem",
            "front angle of a classic Chevrolet Astra five-door hatchback car showing crisp wedge-shaped front hood and lower air dam",
            "rear view of a Chevrolet Astra Opel Astra hatchback car with sharp rectangular corner taillights and angled rear hatch window",
            "rear angle of a Chevrolet Astra medium hatchback showing clean muscular shoulders and compact rear spoiler",
            "a photo of a Chevrolet Astra Opel Astra five-door medium hatchback car in Brazil",
        ],
    },
    "Chevrolet Kadett": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic 1990s Chevrolet Kadett Opel Kadett E two-door hatchback car with sloping aerodynamic nose, rectangular headlights and black or body-colored bumper",
            "front angle of a vintage 1990s Chevrolet Kadett two-door hatchback car on a Brazilian street",
            "rear view of a classic Chevrolet Kadett Opel Kadett E hatchback with distinctive black C-pillar louver vents and small rectangular taillights",
            "rear angle of an older 1990s Chevrolet Kadett two-door wedge hatchback showing covered rear wheel arch skirt and sloping tailgate",
            "a photo of a classic 1990s Chevrolet Kadett Opel Kadett E hatchback car in Brazil",
        ],
    },
    "Chevrolet Agile": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Agile tall compact hatchback car with oversized trapezoidal dual-port grille, large angular headlights and gold Chevrolet bowtie",
            "front angle of a Chevrolet Agile high-roof subcompact hatchback showing tall boxy hood and wide chrome bar across the grille",
            "rear view of a Chevrolet Agile tall hatchback car with high roofline, triangular corner taillights and flat upright tailgate",
            "rear angle of a Chevrolet Agile compact hatchback showing blacked-out C-pillar glass illusion and tall rear bumper",
            "a photo of a Chevrolet Agile tall compact hatchback car on a Brazilian street",
        ],
    },
    "Chevrolet Sonic": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Sonic Aveo hatchback car with exposed twin round motorcycle-style quad headlights, dual-port grille and gold bowtie",
            "front angle of a sporty Chevrolet Sonic compact hatchback showing aggressive open twin round headlamps and sharp hood lines",
            "rear view of a Chevrolet Sonic Aveo hatchback car with hidden rear door handles in C-pillar and twin round-element taillights",
            "rear angle of a Chevrolet Sonic compact hatchback showing upright rear hatch and sporty roof spoiler",
            "a photo of a Chevrolet Sonic Aveo compact hatchback car on a Brazilian road",
        ],
    },
    "Fiat Stilo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Stilo medium five-door hatchback car with sharp rectangular grille, angular crisp headlights and flat sculpted hood",
            "front angle of a Fiat Stilo medium hatchback showing German-inspired crisp straight lines and wide stance",
            "rear view of a Fiat Stilo five-door hatchback car with wide horizontal tail lamps wrapping into the flat tailgate and sharp creases",
            "rear angle of a Fiat Stilo medium hatchback showing boxy muscular rear shoulders and wide C-pillar",
            "a photo of a Fiat Stilo five-door medium hatchback car in Brazil",
        ],
    },
    "Fiat Bravo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Bravo sporty medium hatchback car with sweeping teardrop headlights, rounded trapezoidal grille and red Fiat badge",
            "front angle of a sleek Italian Fiat Bravo five-door hatchback showing raked windshield and aerodynamic nose",
            "rear view of a Fiat Bravo medium hatchback car with distinctive round twin-circle taillights and thick curved C-pillars",
            "rear angle of a Fiat Bravo sporty hatchback showing curved rear glass, integrated spoiler and muscular haunches",
            "a photo of a Fiat Bravo sporty five-door medium hatchback car on a Brazilian road",
        ],
    },
    "Fiat 500": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny retro Fiat 500 Cinquecento city car with round bubble headlights, chrome whisker grille bar and rounded beetle-like hood",
            "front angle of a cute retro Fiat 500 two-door micro hatchback car on a street",
            "rear view of a tiny rounded Fiat 500 micro city car with chrome-ringed vertical oval taillights and curved hatch",
            "rear angle of a retro Fiat 500 Cinquecento city car showing arched roofline and compact rear overhang",
            "a photo of a retro Fiat 500 Cinquecento micro two-door hatchback car in Brazil",
        ],
    },
    "Volkswagen Fusca": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic air-cooled Volkswagen Beetle Fusca with iconic curved round fenders, round headlights, chrome bumper and sloping front hood",
            "front angle of a vintage classic Volkswagen Fusca Beetle car on a Brazilian street",
            "rear view of a classic Volkswagen Beetle Fusca showing rounded engine decklid, cooling louvers, oval taillights and curved rear fenders",
            "rear angle of a Volkswagen Fusca or New Beetle showing iconic arched bubble silhouette and flared wheel arches",
            "a photo of a classic Volkswagen Beetle Fusca passenger car in Brazil",
        ],
    },
    "Volkswagen CrossFox": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Volkswagen CrossFox rugged tall hatchback with black plastic body cladding, silver front skid plate, roof rails and auxiliary fog lights",
            "front angle of a Volkswagen CrossFox adventure compact car showing elevated ride height and rugged black bumper",
            "rear view of a Volkswagen CrossFox adventure hatchback featuring a large external spare tire mounted on a swing-away rear carrier",
            "rear angle of a Volkswagen CrossFox showing high roof rack rails, black wheel arch flares and rear-mounted spare wheel",
            "a photo of a rugged Volkswagen CrossFox adventure compact hatchback car in Brazil",
        ],
    },
    "Nissan March": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny rounded bubble-shaped Nissan Micra K13 March subcompact city car",
            "front angle of a small bubbly Nissan Micra March hatchback car",
            "rear view of a Nissan Micra K13 March rounded subcompact hatchback with curvy round tailgate",
            "rear angle of a small cute bubble-shaped Nissan Micra March city car",
            "a photo of a Nissan Micra K13 March subcompact city hatchback",
        ],
    },
    "Nissan Tiida": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Nissan Tiida C11 Versa hatchback car",
            "front angle of a 2007-2013 Nissan Tiida C11 five-door hatchback car",
            "rear view of a Nissan Tiida C11 hatchback car",
            "rear angle of a Nissan Tiida C11 five-door hatchback car",
            "a photo of a Nissan Tiida C11 Versa five-door hatchback car",
        ],
    },
    "Hyundai i30": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Hyundai i30 medium hatchback car with hexagonal front grille, sleek swept-back headlights and Hyundai emblem",
            "front angle of a sporty Hyundai i30 five-door medium hatchback showing sculpted hood and fog lights",
            "rear view of a Hyundai i30 medium hatchback car with tall vertical pillar taillights or wide wraparound LED lamps and roof spoiler",
            "rear angle of a Hyundai i30 five-door hatchback showing wide stance and aerodynamic rear hatch",
            "a photo of a Hyundai i30 medium five-door hatchback car on a Brazilian road",
        ],
    },
    "Kia Picanto": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny Kia Picanto Morning micro city hatchback with Kia tiger-nose grille, large bug-eyed or angular headlights and short hood",
            "front angle of an ultra-compact Kia Picanto five-door city car on a Brazilian street",
            "rear view of a tiny Kia Picanto micro hatchback with C-shaped vertical corner taillights and flat rear hatch",
            "rear angle of a small Kia Picanto city car showing tall upright stance and very short rear overhang",
            "a photo of a Kia Picanto Morning micro five-door city hatchback car in Brazil",
        ],
    },
    "Kia Soul": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a boxy Kia Soul urban crossover hatchback with upright flat windshield, tiger-nose grille and squared-off front nose",
            "front angle of a distinctive cube-shaped Kia Soul boxy hatchback car on a street",
            "rear view of a boxy Kia Soul car with iconic tall vertical boomerang taillights framing a flat upright rear tailgate",
            "rear angle of a Kia Soul urban box car showing flat vertical rear end and squared roofline",
            "a photo of a boxy Kia Soul five-door urban hatchback car in Brazil",
        ],
    },
    "Chery QQ": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny Chery QQ micro city car with cheerful rounded headlights, smiling front bumper grille and tall narrow body",
            "front angle of an ultra-compact Chery QQ micro hatchback car on a Brazilian street",
            "rear view of a tiny Chery QQ subcompact car with vertical corner taillights, narrow body and flat glass hatch",
            "rear angle of a small narrow Chery QQ city hatchback showing tall upright cabin and small wheels",
            "a photo of a tiny Chery QQ micro city hatchback car in Brazil",
        ],
    },
    "Renault Clio": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a classic Renault Clio subcompact hatchback car with triangular headlights, split grille flanking the Renault diamond and curved hood",
            "front angle of a compact Renault Clio hatchback car on a Brazilian street",
            "rear view of a Renault Clio small hatchback car with compact rounded-triangle corner taillights and curved rear glass",
            "rear angle of a Renault Clio subcompact hatchback showing short two-box silhouette and Renault badge",
            "a photo of a Renault Clio subcompact two-box hatchback car in Brazil",
        ],
    },
    "Renault Megane": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Renault Megane medium car with sharp swept headlights, twin grille openings flanking the Renault diamond and sculpted hood",
            "front angle of a Renault Megane hatchback or sedan passenger car on a Brazilian street",
            "rear view of a Renault Megane car showing distinctive angular bustle-back rear window or sleek horizontal taillights",
            "rear angle of a Renault Megane medium passenger car showing wide sculpted rear quarters and Renault emblem",
            "a photo of a Renault Megane medium passenger car in Brazil",
        ],
    },
    "Suzuki Swift": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a sporty Suzuki Swift subcompact hatchback with flat floating roof, blackened A-pillars, large vertical headlights and Suzuki S grille",
            "front angle of a Suzuki Swift compact hatchback car showing wide stance and sculpted bumper",
            "rear view of a Suzuki Swift hatchback car with chunky corner taillights, blacked-out C-pillar and flat roof",
            "rear angle of a Suzuki Swift mini-inspired hatchback showing short rear overhang and wide rear track",
            "a photo of a Suzuki Swift subcompact hatchback car on a Brazilian road",
        ],
    },
    # --- Sedans adicionados ---
    "Chevrolet Vectra": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Vectra Opel Vectra medium executive sedan with angular headlights, horizontal chrome grille bar and Chevrolet bowtie",
            "front angle of a Chevrolet Vectra four-door sedan showing streamlined side mirrors flowing from the hood or crisp midsize proportions",
            "rear view of a Chevrolet Vectra Opel Vectra four-door sedan with elongated separate trunk lid and trapezoidal corner taillights",
            "rear angle of a Chevrolet Vectra executive three-box sedan showing clean wide rear deck and chrome trim",
            "a photo of a Chevrolet Vectra Opel Vectra four-door medium sedan car in Brazil",
        ],
    },
    "Chevrolet Monza": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a classic 1980s-1990s Chevrolet Monza Opel Ascona C boxy sedan with rectangular headlights, horizontal slatted grille and flat hood",
            "front angle of a vintage boxy Chevrolet Monza classic Brazilian sedan car on a street",
            "rear view of a classic boxy Chevrolet Monza four-door sedan with long flat separate trunk and wide rectangular horizontal taillights",
            "rear angle of a 1990s Chevrolet Monza Tubarao three-box sedan showing sharp angular body lines and chrome window trim",
            "a photo of a classic 1980s-1990s Chevrolet Monza four-door sedan car in Brazil",
        ],
    },
    "Chevrolet Cobalt": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Cobalt compact tall sedan with large dual-port grille, oversized headlights and gold Chevrolet bowtie",
            "front angle of a spacious Chevrolet Cobalt four-door sedan showing tall cabin roof and upright front fascia",
            "rear view of a Chevrolet Cobalt four-door sedan with tall boxy separate rear trunk and vertical or split L-shaped taillights",
            "rear angle of a Chevrolet Cobalt tall three-box family sedan showing high rear decklid and generous cabin height",
            "a photo of a Chevrolet Cobalt four-door compact three-box sedan car on a Brazilian road",
        ],
    },
    "Chevrolet Omega": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large executive Chevrolet Omega Opel Omega Holden Commodore sedan with long hood, wide grille and low luxury stance",
            "front angle of a classic or modern Chevrolet Omega full-size luxury sedan car in Brazil",
            "rear view of a large Chevrolet Omega four-door luxury sedan with long wide rear trunk and classic wraparound taillights",
            "rear angle of a Chevrolet Omega Opel Omega executive rear-wheel-drive three-box sedan showing long wheelbase",
            "a photo of a large executive Chevrolet Omega four-door sedan car in Brazil",
        ],
    },
    "Fiat Linea": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Fiat Linea compact-medium sedan with large chrome-rimmed trapezoidal front grille, Punto-style swept headlights and red Fiat badge",
            "front angle of an elegant Fiat Linea four-door sedan showing chrome grille slats and flowing roofline",
            "rear view of a Fiat Linea four-door sedan with elongated separate rear trunk, chrome strip above the license plate and teardrop taillights",
            "rear angle of a Fiat Linea three-box sedan showing rounded rear quarters and sleek boot lid",
            "a photo of a Fiat Linea four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Fiat Marea": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a classic late-1990s Fiat Marea medium sedan with slim oval grille, almond headlights and curved aerodynamic front hood",
            "front angle of a Fiat Marea four-door passenger sedan on a Brazilian street",
            "rear view of a Fiat Marea four-door sedan with distinctive curved boomerang or triangular rear taillights on a rounded trunk lid",
            "rear angle of a Fiat Marea three-box sedan showing smooth rounded 1990s Italian styling and rear bumper",
            "a photo of a classic Fiat Marea four-door medium sedan car in Brazil",
        ],
    },
    "Fiat Tempra": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a classic 1990s Fiat Tempra boxy sedan with rectangular headlights, narrow horizontal grille and flat angular hood",
            "front angle of a vintage 1990s Fiat Tempra four-door sedan car on a Brazilian street",
            "rear view of a classic 1990s Fiat Tempra sedan with high tall boxy separate trunk and wide dark-tinted horizontal taillights",
            "rear angle of a boxy 1990s Fiat Tempra three-box sedan showing crisp straight beltline and tall rear deck",
            "a photo of a classic 1990s Fiat Tempra four-door three-box sedan car in Brazil",
        ],
    },
    "Ford Fusion": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large Ford Fusion Mondeo executive sedan with prominent Aston Martin trapezoidal chrome grille or three-bar chrome grille and sleek headlights",
            "front angle of a modern Ford Fusion midsize luxury sedan showing long sculpted hood and low wide stance",
            "rear view of a Ford Fusion four-door executive sedan with elongated trunk lid, horizontal LED taillights and dual exhausts",
            "rear angle of a Ford Fusion Mondeo fastback-profile three-box sedan showing chrome trunk bar and Ford oval",
            "a photo of a large executive Ford Fusion four-door sedan car on a Brazilian road",
        ],
    },
    "Honda Accord": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large executive Honda Accord midsize sedan with wide chrome front wing grille, sharp LED headlights and long low hood",
            "front angle of a Honda Accord executive four-door sedan showing long wheelbase and wide luxury stance",
            "rear view of a Honda Accord four-door executive sedan with elongated trunk lid and wide C-shaped or horizontal split taillights",
            "rear angle of a Honda Accord midsize three-box sedan showing sleek fastback roof taper and dual exhaust tips",
            "a photo of a large Honda Accord four-door executive sedan car on a Brazilian highway",
        ],
    },
    "Hyundai Elantra": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Hyundai Elantra Avante medium sedan with sweeping fluidic headlights, hexagonal grille and sleek coupe-like silhouette",
            "front angle of a Hyundai Elantra four-door medium sedan showing sculpted hood creases and Hyundai emblem",
            "rear view of a Hyundai Elantra four-door sedan with elongated trunk lid and sharp sweeping wraparound taillights",
            "rear angle of a Hyundai Elantra three-box sedan showing fastback C-pillar curve and sporty rear bumper",
            "a photo of a Hyundai Elantra medium four-door sedan car on a Brazilian road",
        ],
    },
    "Hyundai Azera": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large luxury Hyundai Azera Grandeur executive sedan with bold vertical-slat or cascading chrome grille and sweeping headlights",
            "front angle of a wide low Hyundai Azera full-size luxury sedan showing long hood and chrome window moldings",
            "rear view of a large Hyundai Azera Grandeur sedan with wide connected full-width taillight bar or sleek wraparound LED lamps and dual exhausts",
            "rear angle of a Hyundai Azera executive three-box luxury sedan showing muscular rear haunches and long decklid",
            "a photo of a large luxury Hyundai Azera Grandeur four-door sedan car in Brazil",
        ],
    },
    "Kia Cerato": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Kia Cerato Forte medium sedan with Kia tiger-nose grille, sharp swept headlights and sporty front bumper",
            "front angle of a Kia Cerato four-door sedan showing low wide stance and sculpted hood",
            "rear view of a Kia Cerato Forte four-door sedan with elongated separate trunk lid and wide horizontal LED taillights",
            "rear angle of a Kia Cerato three-box medium sedan showing coupe-like roofline and integrated trunk lip",
            "a photo of a Kia Cerato Forte four-door medium sedan car on a Brazilian road",
        ],
    },
    "Renault Fluence": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Renault Fluence medium sedan with sweeping curved nose, large Renault chrome diamond and almond headlights",
            "front angle of a Renault Fluence four-door family sedan showing smooth rounded hood and chrome lower grille trim",
            "rear view of a Renault Fluence four-door sedan with large rounded separate trunk and wide horizontal split taillights",
            "rear angle of a Renault Fluence three-box medium sedan showing long rear overhang and curved beltline",
            "a photo of a Renault Fluence four-door medium sedan car on a Brazilian road",
        ],
    },
    "Renault Symbol": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Renault Symbol Clio Sedan compact car with large triangular headlights, horizontal grille bar and Renault diamond emblem",
            "front angle of a Renault Symbol four-door compact sedan on a Brazilian street",
            "rear view of a Renault Symbol four-door compact sedan with tall narrow separate trunk lid and wraparound corner taillights",
            "rear angle of a Renault Symbol three-box compact sedan showing Clio-derived front doors and elongated boot",
            "a photo of a Renault Symbol four-door compact three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Bora": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Bora Jetta Mk4 medium sedan with boxy rectangular headlights, horizontal chrome grille and rounded roof arch",
            "front angle of a classic Volkswagen Bora four-door sedan car on a Brazilian street",
            "rear view of a Volkswagen Bora Jetta Mk4 sedan with tall clean separate trunk lid and square red-and-amber corner taillights",
            "rear angle of a Volkswagen Bora three-box compact-medium sedan showing thick C-pillar and clean German lines",
            "a photo of a Volkswagen Bora four-door three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Passat": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large executive Volkswagen Passat midsize sedan with wide horizontal chrome-slatted grille seamlessly joining crisp headlights",
            "front angle of a Volkswagen Passat luxury four-door sedan showing long wide hood and low executive stance",
            "rear view of a Volkswagen Passat executive sedan with wide clean trunk lid, centered VW emblem and sharp horizontal LED taillights",
            "rear angle of a Volkswagen Passat midsize three-box sedan showing long chrome-trimmed side windows and wide rear track",
            "a photo of a large executive Volkswagen Passat four-door sedan car on a Brazilian highway",
        ],
    },
    "Toyota Camry": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a large luxury Toyota Camry executive sedan with wide lower bumper grille, sleek headlights and prominent Toyota badge",
            "front angle of a Toyota Camry full-size executive four-door sedan showing long hood and wide luxury stance",
            "rear view of a Toyota Camry four-door sedan with wide separate trunk lid and large horizontal split taillights",
            "rear angle of a Toyota Camry executive three-box sedan showing long wheelbase and refined rear bumper",
            "a photo of a large executive Toyota Camry four-door sedan car in Brazil",
        ],
    },
    "Peugeot 408": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Peugeot 408 medium sedan with steeply sloped windshield, large wide front grille, sharp feline headlights and Peugeot lion",
            "front angle of a Peugeot 408 four-door sedan showing long curved hood and chrome grille trim",
            "rear view of a Peugeot 408 four-door sedan with high sculpted trunk lid and wraparound red boomerang taillights",
            "rear angle of a Peugeot 408 three-box medium sedan showing elongated rear quarters and tall boot",
            "a photo of a Peugeot 408 four-door medium sedan car on a Brazilian road",
        ],
    },
    "Citroën C4 Lounge": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Citroen C4 Lounge medium sedan with chrome double chevron grille bars extending into sharp headlights and sculpted hood",
            "front angle of a modern Citroen C4 Lounge four-door sedan showing chrome fog lamp trim and sleek profile",
            "rear view of a Citroen C4 Lounge four-door sedan with concave rear window, distinct trunk lid and wide horizontal LED taillights",
            "rear angle of a Citroen C4 Lounge three-box executive sedan showing chrome window surrounds and double chevron badge",
            "a photo of a Citroen C4 Lounge four-door medium sedan car on a Brazilian road",
        ],
    },
    "Citroën C4 Pallas": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Citroen C4 Pallas medium sedan with boomerang headlights, double chevron chrome grille bars and curved hood",
            "front angle of a Citroen C4 Pallas four-door executive sedan on a Brazilian street",
            "rear view of a Citroen C4 Pallas four-door sedan with long protruding separate trunk and distinctive high L-shaped corner taillights",
            "rear angle of a Citroen C4 Pallas long-wheelbase three-box sedan showing tall vertical trunk edge and curved roofline",
            "a photo of a Citroen C4 Pallas four-door medium sedan car in Brazil",
        ],
    },
    "BYD Seal": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a sleek electric BYD Seal fastback sports sedan with closed aerodynamic nose, boomerang C-shaped LED headlights and BYD badge",
            "front angle of a modern BYD Seal EV sedan showing low aggressive hood, flush door handles and sporty front splitter",
            "rear view of a BYD Seal electric sports sedan with full-width continuous dotted LED light bar and aerodynamic rear diffuser",
            "rear angle of a sleek BYD Seal EV fastback sedan showing sloping coupe roofline and muscular rear fenders",
            "a photo of a modern BYD Seal all-electric sports sedan car on a Brazilian road",
        ],
    },
    "BYD King": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a BYD King Destroyer 05 hybrid sedan with wide parametric dot-matrix grille, slim LED headlights and chrome front bar",
            "front angle of a modern BYD King four-door plug-in hybrid sedan on a Brazilian street",
            "rear view of a BYD King four-door hybrid sedan featuring a sleek full-width continuous LED taillight bar across the trunk",
            "rear angle of a BYD King three-box sedan showing clean fastback C-pillar and modern aerodynamic bumper",
            "a photo of a modern BYD King Destroyer 05 hybrid sedan car on a Brazilian road",
        ],
    },
    # --- Peruas / Station Wagons ---
    "Fiat Palio Weekend": {
        "categoria": "Perua / Station Wagon",
        "prompts": [
            "side and front three-quarters view of a long Fiat Palio Weekend Adventure station wagon estate car showing elongated wagon roofline, long rear cargo window and roof rack rails",
            "three-quarters view of a Fiat Palio Weekend elongated station wagon estate car with plastic wheel arch cladding and long rear overhang",
            "rear view of a Fiat Palio Weekend station wagon estate car with elongated cargo roofline, tall rear quarter windows and wide wagon tailgate",
            "rear angle of a Fiat Palio Weekend elongated estate station wagon showing long rear cargo compartment and roof rails",
            "a photo of a long five-door Fiat Palio Weekend station wagon estate car in Brazil",
        ],
    },
    "Volkswagen Parati": {
        "categoria": "Perua / Station Wagon",
        "prompts": [
            "front three-quarters view of a Volkswagen Parati station wagon estate car with Gol front fascia, elongated roof with rails and long side windows",
            "front angle of a Volkswagen Parati Gol Variant station wagon estate car on a Brazilian street",
            "rear view of a Volkswagen Parati station wagon estate car with long rear cargo area, upright tailgate and horizontal taillights",
            "rear angle of a Volkswagen Parati elongated estate wagon showing three side windows and squared rear overhang",
            "a photo of a Volkswagen Parati station wagon estate car in Brazil",
        ],
    },
    "Volkswagen SpaceFox": {
        "categoria": "Perua / Station Wagon",
        "prompts": [
            "front three-quarters view of a Volkswagen SpaceFox Suran tall station wagon estate car with Fox front headlights, roof rack rails and elongated body",
            "front angle of a Volkswagen SpaceFox tall compact station wagon estate car on a Brazilian street",
            "rear view of a Volkswagen SpaceFox Suran station wagon with tall elongated cargo rear end, long rear side windows and horizontal taillights",
            "rear angle of a Volkswagen SpaceFox tall estate wagon showing vertical rear tailgate and roof rails",
            "a photo of a Volkswagen SpaceFox Suran station wagon estate car in Brazil",
        ],
    },
    "Toyota Fielder": {
        "categoria": "Perua / Station Wagon",
        "prompts": [
            "front three-quarters view of a Toyota Fielder Corolla Fielder station wagon estate car with classic Corolla front grille, headlights and elongated roof with rails",
            "front angle of a Toyota Corolla Fielder five-door station wagon car on a Brazilian street",
            "rear view of a Toyota Fielder station wagon estate car with long extended rear cargo glass, vertical-corner taillights and wide tailgate",
            "rear angle of a Toyota Corolla Fielder estate wagon showing elongated three-window side profile and Toyota badge",
            "a photo of a Toyota Corolla Fielder station wagon estate car in Brazil",
        ],
    },
    # --- SUVs / Crossovers adicionados ---
    "Honda WR-V": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Honda WR-V Fit-based subcompact crossover with short steeply sloped MPV hood, thick chrome grille bar, black cladding and roof rack rails",
            "front angle of a Honda WR-V rugged subcompact tall-roof Fit crossover showing small wheels and silver skid plate",
            "rear view of a Honda WR-V subcompact crossover with L-shaped horizontal taillights wrapping onto the tailgate and tall Fit-like rear glass",
            "rear angle of a Honda WR-V subcompact crossover showing short upright tail, roof rails and raised ride height",
            "a photo of a Honda WR-V Fit-based subcompact crossover car",
        ],
    },
    "Honda CR-V": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large full-bodied Honda CR-V midsize family SUV with wide heavy front bumper, bold Honda grille and large wheelbase",
            "front angle of a large Honda CR-V family SUV showing tall broad hood, wide track and commanding midsize SUV proportions",
            "rear view of a large Honda CR-V midsize SUV featuring signature tall vertical L-shaped D-pillar taillights flanking the wide rear glass",
            "rear angle of a large Honda CR-V family SUV showing long rear quarter windows, wide upright tailgate and black lower cladding",
            "a photo of a large Honda CR-V midsize family SUV on a road",
        ],
    },
    "Jeep Commander": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large 7-seat Jeep Commander SUV with seven-slot front grille, slim rectangular LED headlights, squared hood and roof rack rails",
            "front angle of a long large Jeep Commander three-row SUV showing upright bold front fascia and Jeep lettering",
            "rear view of a large Jeep Commander 7-seater SUV with long boxy rear overhang, flat upright tailgate and ultra-slim horizontal LED taillights",
            "rear angle of a large Jeep Commander SUV showing elongated rear quarter glass, chrome roof arch trim and high ground clearance",
            "a photo of a large 7-seat Jeep Commander SUV car on a Brazilian road",
        ],
    },
    "Volkswagen Taos": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen Taos medium SUV with wide illuminated grille light bar, prominent black front bumper X-trim, LED headlights and roof rails",
            "front angle of a modern Volkswagen Taos compact-midsize SUV showing high sculpted hood and squared wheel arches",
            "rear view of a Volkswagen Taos medium SUV with crisp geometric horizontal LED taillights, centered VW emblem and roof rack rails",
            "rear angle of a modern Volkswagen Taos crossover SUV showing clean squared German proportions and silver lower diffuser",
            "a photo of a modern Volkswagen Taos medium crossover SUV car on a Brazilian road",
        ],
    },
    "Volkswagen Tiguan": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen Tiguan Allspace midsize SUV with wide multi-slat chrome grille, sharp rectangular headlights, roof rails and high ground clearance",
            "front angle of a Volkswagen Tiguan family SUV showing clean sculpted hood creases and prominent VW emblem",
            "rear view of a Volkswagen Tiguan midsize SUV with sharp geometric L-shaped LED taillights, crisp horizontal tailgate creases and roof rails",
            "rear angle of a Volkswagen Tiguan Allspace 7-seat SUV showing long rear overhang and clean muscular wheel arches",
            "a photo of a Volkswagen Tiguan midsize family SUV car on a Brazilian road",
        ],
    },
    "Chevrolet Equinox": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Chevrolet Equinox midsize crossover SUV with wide chrome dual-port grille, sleek headlights, roof rails and gold or black bowtie",
            "front angle of a modern Chevrolet Equinox SUV showing sculpted front fenders and wide commanding stance",
            "rear view of a Chevrolet Equinox midsize SUV with wide horizontal wraparound LED taillights and sculpted rear tailgate",
            "rear angle of a Chevrolet Equinox crossover SUV showing upswept C-pillar window line, roof rails and dual exhausts",
            "a photo of a Chevrolet Equinox midsize crossover SUV car on a Brazilian road",
        ],
    },
    "Chevrolet Trailblazer": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large rugged Chevrolet Trailblazer 7-seat body-on-frame SUV with massive S10-style front grille, high hood, side steps and roof rails",
            "front angle of a tall heavy-duty Chevrolet Trailblazer 4x4 SUV on a Brazilian road",
            "rear view of a large tall Chevrolet Trailblazer SUV with upright rear glass, high ground clearance and wide horizontal taillights",
            "rear angle of a big Chevrolet Trailblazer 4x4 SUV showing side running boards, long rear side windows and roof rack rails",
            "a photo of a large rugged Chevrolet Trailblazer 7-seat 4x4 SUV in Brazil",
        ],
    },
    "Hyundai ix35": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Hyundai ix35 Tucson LM crossover SUV with hexagonal grille, swept-back fluidic headlights, roof rails and sculpted hood",
            "front angle of a Hyundai ix35 compact-medium SUV on a Brazilian road showing dynamic curvy front bumper",
            "rear view of a Hyundai ix35 crossover SUV with sloped rear window, wraparound leaf-shaped taillights and black lower bumper cladding",
            "rear angle of a Hyundai ix35 fluidic crossover SUV showing upswept rear window beltline and roof rack rails",
            "a photo of a Hyundai ix35 crossover SUV car on a Brazilian road",
        ],
    },
    "Hyundai Santa Fe": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large Hyundai Santa Fe midsize-large SUV with wide chrome hexagonal or cascading grille, muscular hood and roof rack rails",
            "front angle of a Hyundai Santa Fe 7-seat family SUV showing wide commanding stance and high ground clearance",
            "rear view of a large Hyundai Santa Fe SUV with wide wraparound taillights, dual exhausts and tall rear tailgate",
            "rear angle of a Hyundai Santa Fe family SUV showing long three-row body, roof rails and silver lower skid plate",
            "a photo of a large Hyundai Santa Fe family SUV car on a Brazilian road",
        ],
    },
    "Kia Sportage": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Kia Sportage medium crossover SUV with Kia tiger-nose grille, high-mounted sleek headlights or boomerang DRLs and roof rack rails",
            "front angle of a Kia Sportage crossover SUV showing bold muscular front fascia and black wheel arch cladding",
            "rear view of a Kia Sportage medium SUV with wide horizontal connected taillights and sculpted high rear bumper",
            "rear angle of a Kia Sportage crossover SUV showing thick sporty C-pillar, roof spoiler and elevated ride height",
            "a photo of a Kia Sportage medium crossover SUV car on a Brazilian road",
        ],
    },
    "Kia Sorento": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large 7-seat Kia Sorento family SUV with wide tiger-nose grille, squared muscular hood and roof rack rails",
            "front angle of a large Kia Sorento midsize-large SUV showing commanding height and black lower cladding",
            "rear view of a large Kia Sorento SUV with tall upright tailgate and wide horizontal or split vertical LED taillights",
            "rear angle of a large Kia Sorento three-row SUV showing long squared body, roof rails and high ground clearance",
            "a photo of a large Kia Sorento family SUV car on a Brazilian road",
        ],
    },
    "Mitsubishi ASX": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Mitsubishi ASX Outlander Sport RVR compact crossover SUV with bold trapezoidal jet-fighter grille or Dynamic Shield chrome X-face",
            "front angle of a Mitsubishi ASX compact SUV showing sharp angry headlights, black bumper cladding and roof rails",
            "rear view of a Mitsubishi ASX compact crossover SUV with sharp slim horizontal taillights and angled rear hatch",
            "rear angle of a Mitsubishi ASX RVR crossover SUV showing compact muscular stance, roof rails and high ground clearance",
            "a photo of a Mitsubishi ASX Outlander Sport compact crossover SUV car in Brazil",
        ],
    },
    "Mitsubishi Outlander": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Mitsubishi Outlander medium-large SUV with Dynamic Shield chrome X-frame front grille, slim headlights and roof rack rails",
            "front angle of a Mitsubishi Outlander 7-seat family SUV showing long hood and elevated ground clearance",
            "rear view of a Mitsubishi Outlander SUV with long upright rear tailgate, silver roof rails and horizontal LED taillights",
            "rear angle of a Mitsubishi Outlander family SUV showing elongated rear quarter windows and clean boxy rear profile",
            "a photo of a Mitsubishi Outlander midsize-large SUV car on a Brazilian road",
        ],
    },
    "Mitsubishi Pajero": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a rugged Mitsubishi Pajero Montero Sport Full TR4 4x4 SUV with tall ground clearance, bold chrome grille, side steps and roof rack rails",
            "front angle of a heavy-duty Mitsubishi Pajero off-road 4x4 SUV on a Brazilian road",
            "rear view of a Mitsubishi Pajero SUV with tall boxy upright body, high clearance, vertical or dripping taillights or external rear spare tire",
            "rear angle of a rugged Mitsubishi Pajero Montero 4x4 SUV showing flared wheel arches, running boards and roof rails",
            "a photo of a rugged Mitsubishi Pajero Montero 4x4 off-road SUV in Brazil",
        ],
    },
    "Toyota RAV4": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Toyota RAV4 medium SUV with bold octagonal grille, angular headlights, squared black wheel arches and roof rack rails",
            "front angle of a Toyota RAV4 crossover SUV showing high sculpted hood and commanding stance",
            "rear view of a Toyota RAV4 SUV with wide horizontal taillights connected by a chrome or black trim bar and upright rear tailgate",
            "rear angle of a Toyota RAV4 crossover SUV showing trapezoidal fender flares, roof rails and high ground clearance",
            "a photo of a Toyota RAV4 medium crossover SUV car on a Brazilian road",
        ],
    },
    "Peugeot 2008": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Peugeot 2008 compact crossover SUV with lion-claw headlights, bold front grille, black wheel arch cladding and roof rack rails",
            "front angle of a modern Peugeot 2008 compact SUV showing sculpted hood and Peugeot lion badge",
            "rear view of a Peugeot 2008 compact SUV with glossy black horizontal band connecting three-claw LED taillights and black rear bumper",
            "rear angle of a Peugeot 2008 crossover SUV showing floating roof accents, roof rails and raised ground clearance",
            "a photo of a Peugeot 2008 compact crossover SUV car on a Brazilian road",
        ],
    },
    "Peugeot 3008": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a modern Peugeot 3008 medium SUV with frameless parametric grille or saber-tooth LED DRLs, tall muscular hood and roof rails",
            "front angle of a Peugeot 3008 crossover SUV showing sleek floating roof, chrome trim and high ground clearance",
            "rear view of a Peugeot 3008 medium SUV with full-width black band housing three-claw LED taillights and high sculpted bumper",
            "rear angle of a Peugeot 3008 SUV showing sharp modern creases, blacked-out D-pillar and roof rails",
            "a photo of a modern Peugeot 3008 medium crossover SUV car on a Brazilian road",
        ],
    },
    "Citroën C4 Cactus": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Citroen C4 Cactus compact crossover SUV with two-tier split LED headlights, double chevron grille, roof rails and protective side Airbumps",
            "front angle of a Citroen C4 Cactus crossover SUV showing high rounded hood and colorful fog lamp bezels",
            "rear view of a Citroen C4 Cactus compact SUV with 3D horizontal taillights, black lower cladding and floating C-pillar",
            "rear angle of a Citroen C4 Cactus crossover showing distinctive lower door Airbump panels and roof rack rails",
            "a photo of a Citroen C4 Cactus compact crossover SUV car on a Brazilian road",
        ],
    },
    "Renault Captur": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Renault Captur Kaptur compact SUV with large Renault chrome diamond on glossy black grille, C-shaped LED DRLs and two-tone roof",
            "front angle of a Renault Captur crossover SUV showing curvy muscular fenders and high ground clearance",
            "rear view of a Renault Captur SUV with rounded tailgate, horizontal C-shaped taillights and black bumper cladding",
            "rear angle of a Renault Captur two-tone crossover SUV showing rounded curves, roof rails and elevated stance",
            "a photo of a Renault Captur compact crossover SUV car on a Brazilian road",
        ],
    },
    "Renault Kardian": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a modern Renault Kardian compact crossover SUV with split headlights, diamond-pattern black grille, flat Renault logo and roof rack rails",
            "front angle of a modern Renault Kardian subcompact SUV showing sculpted hood, silver skid plate and black wheel arch cladding",
            "rear view of a Renault Kardian compact SUV with distinctive C-shaped boomerang taillights and upright sculpted tailgate",
            "rear angle of a modern Renault Kardian crossover SUV showing modular roof rails and high ground clearance",
            "a photo of a modern Renault Kardian compact crossover SUV car on a Brazilian road",
        ],
    },
    "Chery Tiggo 5x": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Chery Tiggo 5x SUV",
            "front angle of a Chery Tiggo 4 Tiggo 5x crossover SUV",
            "rear view of a Chery Tiggo 5x SUV",
            "rear angle of a Chery Tiggo 4 Tiggo 5x crossover SUV",
            "a photo of a Chery Tiggo 5x SUV car",
        ],
    },
    "Chery Tiggo 7": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Chery Tiggo 7 Pro SUV",
            "front angle of a Chery Tiggo 7 crossover SUV",
            "rear view of a Chery Tiggo 7 Pro SUV",
            "rear angle of a Chery Tiggo 7 crossover SUV",
            "a photo of a Chery Tiggo 7 Pro SUV car",
        ],
    },
    "Chery Tiggo 8": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large 7-seat Caoa Chery Tiggo 8 SUV with massive diamond-matrix chrome grille, slim LED headlights and roof rack rails",
            "front angle of a large Chery Tiggo 8 three-row family SUV showing tall commanding front fascia and chrome trim",
            "rear view of a large Chery Tiggo 8 7-seat SUV with full-width connected LED light bar, dual exhausts and tall upright tailgate",
            "rear angle of a large Chery Tiggo 8 SUV showing long three-row rear quarter window, roof rails and high ground clearance",
            "a photo of a large 7-seat Caoa Chery Tiggo 8 SUV car on a Brazilian road",
        ],
    },
    "BYD Song Plus": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a BYD Song Plus DM-i Seal U medium SUV with Dragon Face grille or sleek ocean nose, slim LED headlights and roof rack rails",
            "front angle of a modern BYD Song Plus hybrid crossover SUV showing sculpted hood and wide aerodynamic stance",
            "rear view of a BYD Song Plus Seal U medium SUV featuring a sleek full-width continuous LED taillight bar and BYD lettering",
            "rear angle of a BYD Song Plus crossover SUV showing smooth modern body lines, roof rails and tailgate spoiler",
            "a photo of a modern BYD Song Plus DM-i medium SUV car on a Brazilian road",
        ],
    },
    "BYD Yuan Plus": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a BYD Yuan Plus Atto 3 electric compact SUV with thick silver Dragon Face mustache bar across closed grille and LED headlights",
            "front angle of a BYD Yuan Plus Atto 3 EV crossover SUV showing smooth aerodynamic hood and black wheel arch cladding",
            "rear view of a BYD Yuan Plus Atto 3 electric SUV featuring a full-width continuous LED light bar and textured silver dragon-scale C-pillar panel",
            "rear angle of a BYD Yuan Plus Atto 3 EV crossover showing distinctive patterned C-pillar trim and roof rack rails",
            "a photo of a modern BYD Yuan Plus Atto 3 electric crossover SUV car on a Brazilian road",
        ],
    },
    "GWM Haval H6": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a GWM Haval H6 medium hybrid SUV with large cascading star-matrix or horizontal-bar chrome grille, sharp LED headlights and roof rails",
            "front angle of a modern GWM Haval H6 SUV showing tall muscular front fascia, vertical fog lamp intakes and Haval lettering",
            "rear view of a GWM Haval H6 medium SUV featuring a wide continuous full-width LED taillight bar across the tailgate with HAVAL badge",
            "rear angle of a GWM Haval H6 crossover SUV showing sleek coupe-like or boxy rear quarters, roof rails and high ground clearance",
            "a photo of a modern GWM Haval H6 hybrid medium SUV car on a Brazilian road",
        ],
    },
    # --- Picapes adicionadas ---
    "Mitsubishi L200 Triton": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Mitsubishi L200 Triton double-cab 4x4 pickup truck with Dynamic Shield or curved J-line cab-to-bed design and high ground clearance",
            "front angle of a rugged Mitsubishi L200 Triton midsize pickup truck showing bold grille, skid plate and flared wheel arches",
            "rear view of a Mitsubishi L200 Triton pickup truck showing open rear cargo bed, vertical corner taillights and tailgate",
            "rear angle of a Mitsubishi L200 Triton double-cab pickup truck showing open cargo bed, sports bar roll bar and side steps",
            "a photo of a rugged Mitsubishi L200 Triton double-cab 4x4 pickup truck on a Brazilian road",
        ],
    },
    "Nissan Frontier": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Nissan Frontier Navara double-cab 4x4 pickup truck with bold interlocked V-motion grille, C-shaped LED headlights and high hood",
            "front angle of a rugged Nissan Frontier midsize pickup truck showing muscular fenders and commanding off-road stance",
            "rear view of a Nissan Frontier Navara pickup truck showing open rear cargo bed, vertical taillights and stamped tailgate",
            "rear angle of a Nissan Frontier double-cab pickup truck showing open cargo bed, tubular sports bar and side running boards",
            "a photo of a rugged Nissan Frontier Navara double-cab 4x4 pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet D20": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a classic vintage boxy 1980s-1990s Chevrolet D20 pickup truck with square dual or single rectangular headlights and wide chrome slatted grille",
            "front angle of a classic heavy-duty Chevrolet D20 Brazilian pickup truck on a street",
            "rear view of a classic boxy Chevrolet D20 pickup truck showing long straight open steel cargo bed and vertical rectangular taillights",
            "rear angle of an older square Chevrolet D20 single-cab or double-cab pickup truck showing flat 1980s sheet metal and chrome bumper",
            "a photo of a classic vintage boxy Chevrolet D20 pickup truck in Brazil",
        ],
    },
    "Ford Courier": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a small Ford Courier Brazilian compact utility pickup truck with Fiesta front headlights, oval grille and compact cab",
            "front angle of a small Ford Courier two-door light utility pickup truck on a Brazilian street",
            "rear view of a small Ford Courier utility pickup truck showing deep long open rear cargo bed and tall vertical corner taillights",
            "rear angle of a compact Ford Courier single-cab pickup truck showing Fiesta-derived cab attached to a boxy utility cargo bed",
            "a photo of a small Ford Courier Brazilian compact utility pickup truck in Brazil",
        ],
    },
    "Ford Maverick": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a modern Ford Maverick compact unibody pickup truck with wide horizontal bar connecting C-clamp headlights and black lower cladding",
            "front angle of a modern Ford Maverick four-door crew-cab pickup truck showing boxy upright hood and Ford blue oval",
            "rear view of a Ford Maverick pickup truck showing open rear cargo bed, vertical taillights and tailgate with stamped MAVERICK lettering",
            "rear angle of a modern Ford Maverick unibody crew-cab pickup truck showing seamless cab-to-bed side profile and black bed rails",
            "a photo of a modern Ford Maverick compact crew-cab pickup truck on a Brazilian road",
        ],
    },
    "RAM Rampage": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a RAM Rampage pickup truck with massive bold RAM lettering grille, aggressive rectangular LED headlights and muscular hood bulge",
            "front angle of a modern RAM Rampage crew-cab pickup truck showing rugged American truck styling in compact-midsize proportions",
            "rear view of a RAM Rampage pickup truck showing open cargo bed, American-flag patterned vertical LED taillights and large RAM ram-head emblem",
            "rear angle of a RAM Rampage four-door pickup truck showing sculpted rear fenders, dual exhausts and high beltline",
            "a photo of a modern RAM Rampage crew-cab pickup truck on a Brazilian road",
        ],
    },
    "Fiat Titano": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Fiat Titano Peugeot Landtrek double-cab midsize pickup truck with large grille, fang-like vertical LED DRLs and FIAT lettering",
            "front angle of a modern Fiat Titano 4x4 midsize pickup truck showing high hood, silver skid plate and rugged bumper",
            "rear view of a Fiat Titano pickup truck showing open rear cargo bed, vertical corner taillights and wide tailgate with FIAT script",
            "rear angle of a Fiat Titano double-cab pickup truck showing open cargo bed, sports bar roll bar and side steps",
            "a photo of a modern Fiat Titano double-cab midsize pickup truck on a Brazilian road",
        ],
    },
    # --- Minivans / Vans / Utilitários adicionados ---
    "Chevrolet Meriva": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Chevrolet Meriva Opel Meriva A compact tall MPV minivan with short sloping hood, triangular headlights and Chevrolet bowtie",
            "front angle of a Chevrolet Meriva tall one-box compact family minivan on a Brazilian street",
            "rear view of a Chevrolet Meriva Opel Meriva compact minivan with high-mounted vertical taillights flanking the rear hatch glass",
            "rear angle of a Chevrolet Meriva compact MPV showing tall boxy cabin, large side windows and flat rear tailgate",
            "a photo of a Chevrolet Meriva Opel Meriva compact family minivan MPV car in Brazil",
        ],
    },
    "Chevrolet Zafira": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Chevrolet Zafira Opel Zafira A 7-seat family minivan MPV with wide Astra-style headlights, short hood and tall cabin",
            "front angle of a Chevrolet Zafira multi-purpose 7-seater family minivan on a Brazilian street",
            "rear view of a Chevrolet Zafira Opel Zafira MPV minivan with upright flat tailgate, roof rails and horizontal-vertical corner taillights",
            "rear angle of a long Chevrolet Zafira 7-seat family minivan showing three large side windows and tall roof",
            "a photo of a Chevrolet Zafira Opel Zafira 7-seat family minivan MPV car in Brazil",
        ],
    },
    "Fiat Idea": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Fiat Idea compact tall MPV minivan with high upright windshield, large triangular or rounded headlights and tall cabin",
            "front angle of a Fiat Idea family mini-MPV car on a Brazilian street",
            "rear view of a Fiat Idea compact tall minivan with high upright tailgate and tall vertical rear corner taillights",
            "rear angle of a Fiat Idea compact MPV showing tall boxy greenhouse, roof rails and short hood",
            "a photo of a Fiat Idea compact tall family MPV minivan car in Brazil",
        ],
    },
    "Fiat Doblo": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a boxy tall Fiat Doblo utility van and MPV with high flat roof, sliding side doors and tall upright windshield",
            "front angle of a Fiat Doblo Adventure or Cargo tall utility van on a Brazilian street",
            "rear view of a boxy tall Fiat Doblo utility van showing massive tall flat vertical rear hatch or asymmetric split cargo doors and vertical taillights",
            "rear angle of a Fiat Doblo tall cubic utility van showing very high squared roofline and sliding door rail",
            "a photo of a boxy tall Fiat Doblo utility passenger or cargo van in Brazil",
        ],
    },
    "Nissan Livina": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Nissan Livina Grand Livina compact family minivan MPV with split chrome-tooth Nissan grille, large headlights and tall station-wagon cabin",
            "front angle of a Nissan Livina MPV family car on a Brazilian street",
            "rear view of a Nissan Livina compact family minivan with large wraparound horizontal-triangular taillights and tall upright hatch",
            "rear angle of a Nissan Livina MPV minivan showing long roofline, roof rails and spacious rear cabin",
            "a photo of a Nissan Livina compact family minivan MPV car in Brazil",
        ],
    },
    "Volkswagen Kombi": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a classic Volkswagen Kombi Type 2 T2 boxy flat-front van bus with round headlights, prominent round VW emblem on flat nose and split or curved windshield",
            "front angle of a classic white or two-tone Volkswagen Kombi T2 bus van on a Brazilian street",
            "rear view of a classic boxy Volkswagen Kombi T2 van showing flat vertical rear engine hatch, cooling vents and small rectangular taillights",
            "rear angle of a classic Volkswagen Kombi van bus showing iconic boxy bread-loaf silhouette and flat vertical sides",
            "a photo of a classic Volkswagen Kombi Type 2 utility passenger or cargo van in Brazil",
        ],
    },
    "Renault Kangoo": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a Renault Kangoo compact utility cargo or passenger van with short sloping nose, tall high-roof cargo box and Renault diamond",
            "front angle of a white Renault Kangoo small commercial utility panel van on a Brazilian street",
            "rear view of a Renault Kangoo compact high-roof delivery van showing asymmetric split rear cargo doors and tall vertical pillar taillights",
            "rear angle of a Renault Kangoo compact commercial van showing stepped high roof behind the front seats and sliding side door",
            "a photo of a Renault Kangoo compact commercial utility van in Brazil",
        ],
    },
    "Renault Master": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a large tall white Renault Master full-size commercial cargo or passenger van with huge vertical headlights, wide grille and Renault diamond",
            "front angle of a large Renault Master commercial panel van or minibus on a Brazilian road",
            "rear view of a large tall Renault Master full-size van showing tall vertical rear cargo doors, high roof and tall vertical taillights",
            "rear angle of a full-size Renault Master cargo van or executive passenger minibus showing long boxy body",
            "a photo of a large Renault Master commercial delivery van or passenger van in Brazil",
        ],
    },
    "Mercedes-Benz Sprinter": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a large tall Mercedes-Benz Sprinter commercial van or passenger minibus with prominent Mercedes three-pointed star grille and tall hood",
            "front angle of a white Mercedes-Benz Sprinter full-size cargo van or executive passenger van on a Brazilian road",
            "rear view of a large tall Mercedes-Benz Sprinter van showing tall vertical twin rear doors, high roof and vertical corner taillights",
            "rear angle of a full-size Mercedes-Benz Sprinter panel van or minibus showing long boxy high-roof silhouette",
            "a photo of a large Mercedes-Benz Sprinter commercial cargo or passenger van in Brazil",
        ],
    },
    "Fiat Ducato": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a large tall Fiat Ducato full-size commercial cargo van or minibus with short snub nose, high-mounted swept headlights and wide grille",
            "front angle of a white Fiat Ducato commercial panel van or passenger van on a Brazilian road",
            "rear view of a large tall Fiat Ducato full-size van showing tall flat vertical twin rear doors and vertical corner taillights",
            "rear angle of a full-size Fiat Ducato cargo van or executive minibus showing massive boxy high-roof body",
            "a photo of a large Fiat Ducato commercial delivery or passenger van in Brazil",
        ],
    },
    "Citroën Jumper": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a large tall Citroen Jumper full-size commercial cargo van or minibus with double chevron front grille and high-mounted headlights",
            "front angle of a white Citroen Jumper commercial panel van or passenger van on a Brazilian road",
            "rear view of a large tall Citroen Jumper full-size van showing tall vertical twin rear doors and vertical corner taillights",
            "rear angle of a full-size Citroen Jumper cargo van showing long boxy high-roof silhouette",
            "a photo of a large Citroen Jumper commercial utility van in Brazil",
        ],
    },
    "Peugeot Boxer": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a large tall Peugeot Boxer full-size commercial cargo van or minibus with Peugeot lion emblem on wide front grille and high headlights",
            "front angle of a white Peugeot Boxer commercial panel van or passenger van on a Brazilian road",
            "rear view of a large tall Peugeot Boxer full-size van showing tall vertical twin rear doors and vertical corner taillights",
            "rear angle of a full-size Peugeot Boxer cargo van showing long boxy high-roof utility body",
            "a photo of a large Peugeot Boxer commercial utility van in Brazil",
        ],
    },
}

# Prompts Zero-Shot para categorias de carroceria (usados para validação cruzada ou fallback)
PROMPTS_CATEGORIAS = {
    "Hatchback": [
        "a compact hatchback car with a short rear overhang and rear hatch door",
        "rear view of a small two-box hatchback passenger car",
        "front view of a compact two-box hatchback passenger car",
    ],
    "Sedan": [
        "a three-box sedan passenger car with an elongated rear trunk lid",
        "rear view of a classic four-door sedan car with a separate trunk",
        "front view of a four-door sedan car with front hood and grille",
    ],
    "Perua / Station Wagon": [
        "a station wagon estate car with elongated roof and rear cargo window",
        "rear view of a station wagon family estate car",
        "front view of a station wagon passenger car",
    ],
    "SUV / Utilitário": [
        "a sport utility vehicle SUV crossover with high ground clearance and large wheels",
        "rear view of a tall SUV crossover vehicle",
        "front view of a tall SUV crossover vehicle with high ground clearance",
    ],
    "Picape": [
        "a pickup truck with an open cargo bed in the back",
        "rear view of a pickup truck tailgate and cargo bed",
        "front view of a rugged pickup truck showing grille and headlights",
    ],
    "Minivan": [
        "a long multi-purpose family minivan MPV passenger van",
        "front view of a family minivan MPV passenger car",
    ],
    "Furgão / Van": [
        "a commercial cargo delivery van utility panel van",
        "front view of a commercial delivery panel van",
    ],
}

# Mapeamento auxiliar por substring (caso use modelo YOLO supervisionado)
_TOKENS_AMBIGUOS = {
    "plus", "mini", "cross", "novo", "antigo", "rocam", "grand", "sport6",
    "5x", "h6", "7", "8", "500", "206", "207", "307", "308", "408", "2008", "3008",
}
MAPA_MODELOS_CATEGORIA = {}
for _nome, _info in CATALOGO_FROTA_BR.items():
    _base = _nome.split("(")[0].strip().lower()
    _partes = _base.split()
    _cat = _info["categoria"]
    # Nome completo do catálogo (ex: "chevrolet astra", "fiat palio weekend")
    MAPA_MODELOS_CATEGORIA[_base] = _cat
    # Nome do modelo sem a montadora (ex: "palio weekend", "song plus", "tiggo 8", "c4 lounge")
    if len(_partes) >= 2:
        _sem_marca = " ".join(_partes[1:])
        if len(_sem_marca) >= 2 and not _sem_marca.isdigit():
            MAPA_MODELOS_CATEGORIA[_sem_marca] = _cat
    # Última palavra apenas se não for curta ou numérica/ambígua
    _ultimo = _partes[-1]
    if len(_ultimo) >= 3 and not _ultimo.isdigit() and _ultimo not in _TOKENS_AMBIGUOS:
        MAPA_MODELOS_CATEGORIA[_ultimo] = _cat

MAPA_MODELOS_CATEGORIA.update({
    "fox": "Hatchback", "gol": "Hatchback", "palio": "Hatchback", "uno": "Hatchback",
    "uno mille": "Hatchback", "fiesta": "Hatchback", "fiesta rocam": "Hatchback",
    "new fiesta": "Hatchback", "ka": "Hatchback", "up": "Hatchback", "qq": "Hatchback",
    "onix": "Hatchback", "hb20": "Hatchback", "polo": "Hatchback", "celta": "Hatchback",
    "astra": "Hatchback", "march": "Hatchback", "fiat 500": "Hatchback",
    "c4": "Hatchback", "c3": "Hatchback",
    "corolla": "Sedan", "civic": "Sedan", "voyage": "Sedan", "siena": "Sedan",
    "grand siena": "Sedan", "prisma": "Sedan", "virtus": "Sedan", "hb20s": "Sedan",
    "onix plus": "Sedan", "c4 lounge": "Sedan", "c4 pallas": "Sedan",
    "palio weekend": "Perua / Station Wagon", "parati": "Perua / Station Wagon",
    "spacefox": "Perua / Station Wagon", "fielder": "Perua / Station Wagon",
    "compass": "SUV / Utilitário", "renegade": "SUV / Utilitário", "creta": "SUV / Utilitário",
    "t-cross": "SUV / Utilitário", "tracker": "SUV / Utilitário", "kicks": "SUV / Utilitário",
    "corolla cross": "SUV / Utilitário", "c3 aircross": "SUV / Utilitário",
    "c4 cactus": "SUV / Utilitário", "song plus": "SUV / Utilitário",
    "yuan plus": "SUV / Utilitário", "haval h6": "SUV / Utilitário",
    "tiggo 5x": "SUV / Utilitário", "tiggo 7": "SUV / Utilitário", "tiggo 8": "SUV / Utilitário",
    "strada": "Picape", "toro": "Picape", "saveiro": "Picape", "hilux": "Picape", "s10": "Picape",
    "l200 triton": "Picape", "l200": "Picape",
    "spin": "Minivan", "meriva": "Minivan", "zafira": "Minivan", "idea": "Minivan", "livina": "Minivan",
    "fiorino": "Furgão / Van", "kombi": "Furgão / Van", "doblo": "Furgão / Van",
})


def inferir_categoria_por_modelo(nome_modelo: str) -> str:
    """Inferir a carroceria/categoria a partir do nome ou modelo do veículo."""
    if nome_modelo in CATALOGO_FROTA_BR:
        return CATALOGO_FROTA_BR[nome_modelo]["categoria"]
    nome_norm = nome_modelo.strip().lower()
    for k, v in CATALOGO_FROTA_BR.items():
        if k.lower() == nome_norm:
            return v["categoria"]
    for chave, cat in sorted(
        MAPA_MODELOS_CATEGORIA.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if chave in nome_norm:
            return cat
    return "Carro"


def formatar_nome_veiculo(classe: str) -> str:
    """
    Formata strings como 'vw_fox', 'chevrolet-onix', 'volkswagen_fox_2013'
    para 'Volkswagen Fox', 'Chevrolet Onix', etc.
    """
    limpo = classe.replace("-", " ").replace("_", " ")
    substituicoes = {
        "vw": "Volkswagen",
        "gm": "Chevrolet",
        "chevrolet": "Chevrolet",
        "volkswagen": "Volkswagen",
        "fiat": "Fiat",
        "ford": "Ford",
        "hyundai": "Hyundai",
        "toyota": "Toyota",
        "honda": "Honda",
        "renault": "Renault",
        "nissan": "Nissan",
        "jeep": "Jeep",
        "peugeot": "Peugeot",
        "citroen": "Citroën",
        "citroën": "Citroën",
        "mitsubishi": "Mitsubishi",
    }
    partes = limpo.split()
    partes_fmt = []
    for p in partes:
        p_low = p.lower()
        if p_low in substituicoes:
            partes_fmt.append(substituicoes[p_low])
        elif p.isdigit() and len(p) == 4:
            continue
        else:
            partes_fmt.append(p.capitalize())
    return " ".join(partes_fmt).strip() or classe


def estimar_cor_lataria(crop_veiculo: np.ndarray) -> str:
    """
    Analisa a região representativa da lataria do veículo no espaço de cores HSV
    (com correção de reflexo azulado por chroma e estatísticas robustas por mediana/percentil)
    e retorna a cor predominante em português.
    """
    if crop_veiculo is None or crop_veiculo.size == 0:
        return "Indefinida"

    h, w = crop_veiculo.shape[:2]
    if h < 20 or w < 20:
        return "Indefinida"

    # Foca na faixa central/inferior da lataria (evita o terço superior com vidros/céu
    # e o rodapé com pneus/asfalto).
    y1, y2 = int(h * 0.42), int(h * 0.78)
    x1, x2 = int(w * 0.18), int(w * 0.82)
    lataria = crop_veiculo[y1:y2, x1:x2]

    if lataria.size == 0:
        lataria = crop_veiculo

    hsv = cv2.cvtColor(lataria, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Filtra sombras extremas (V < 25) e brilhos estourados (V > 245)
    mascara = (V >= 25) & (V <= 245)
    if np.sum(mascara) < 50:
        mascara = np.ones(V.shape, dtype=bool)

    h_vals = H[mascara]
    s_vals = S[mascara]
    v_vals = V[mascara]

    # Chroma absoluta (max(B,G,R) - min(B,G,R)): impede que reflexos frios/azulados
    # em carros escuros, pratas ou brancos (onde V é baixo e S infla) virem "Azul".
    lataria_m = lataria[mascara].astype(np.int16)
    chroma_vals = lataria_m.max(axis=1) - lataria_m.min(axis=1)

    s_med = float(np.median(s_vals))
    s_p75 = float(np.percentile(s_vals, 75))
    v_med = float(np.median(v_vals))
    c_med = float(np.median(chroma_vals))

    # Matiz circular mediano nos pixels mais saturados (evita wrap-around 0/179 no vermelho)
    mask_sat_h = (s_vals >= 80) & (chroma_vals >= 20)
    h_ref = h_vals[mask_sat_h] if np.sum(mask_sat_h) >= 20 else h_vals
    angulos = h_ref.astype(np.float64) * (2.0 * np.pi / 180.0)
    sin_med = float(np.median(np.sin(angulos)))
    cos_med = float(np.median(np.cos(angulos)))
    if abs(sin_med) < 1e-6 and abs(cos_med) < 1e-6:
        h_med = float(np.median(h_ref))
    else:
        ang = float(np.arctan2(sin_med, cos_med))
        if ang < 0:
            ang += 2.0 * np.pi
        h_med = ang * (180.0 / (2.0 * np.pi))

    # Critério conjunto de cromaticidade: exige saturação + separação real de canais
    # ou saturação alta consistente (ex.: azul/vermelho escuros reais).
    eh_cromatico = (
        (c_med >= 42 and s_med >= 85)
        or (s_med >= 155 and c_med >= 22)
        or (s_p75 >= 185 and s_med >= 100 and c_med >= 14)
    )

    # Cores acromáticas (Branco, Preto, Prata / Cinza)
    if not eh_cromatico:
        if v_med < 55:
            return "Preto"
        elif v_med > 145:
            return "Branco"
        else:
            return "Prata / Cinza"

    # Cores cromáticas baseadas no Matiz (Hue no OpenCV vai de 0 a 179)
    if h_med < 8 or h_med >= 165:
        return "Vermelho"
    elif 8 <= h_med < 26:
        if v_med < 120:
            return "Marrom"
        return "Amarelo / Laranja"
    elif 26 <= h_med < 35:
        return "Amarelo / Laranja"
    elif 35 <= h_med < 85:
        return "Verde"
    elif 85 <= h_med < 135:
        return "Azul"
    elif 135 <= h_med < 165:
        return "Roxo / Vinho"

    return "Outra"


def classificar_tipo_veiculo(cls_id: int) -> str:
    """Retorna o tipo de veículo em português a partir do ID do COCO."""
    return MAPA_TIPOS_VEICULO.get(cls_id, "Carro")


def calcular_matriz_confiabilidade(prob_1: float, prob_2: float = 0.0) -> str:
    """
    Calcula a Matriz de Confiabilidade da predição VMMR baseado na probabilidade
    do 1º colocado e na margem de separação em relação ao 2º colocado:
    - Alta: prob_1 >= 0.40 e margem (prob_1 - prob_2) >= 0.25
    - Média: margem (prob_1 - prob_2) >= 0.10
    - Baixa / Ambígua: margem estreita (< 0.10) ou baixa certeza
    """
    conf_alta_min = getattr(config, "VMMR_CONF_ALTA_MIN", 0.40)
    margem_alta = getattr(config, "VMMR_MARGEM_ALTA", 0.25)
    margem_media = getattr(config, "VMMR_MARGEM_MEDIA", 0.10)

    margem = prob_1 - prob_2
    if prob_1 >= conf_alta_min and margem >= margem_alta:
        return "Alta"
    elif margem >= margem_media:
        return "Média"
    else:
        return "Baixa / Ambígua"


def _extrair_tensor_embedding(saida_modelo) -> torch.Tensor:
    """Garante compatibilidade entre diferentes versões do HuggingFace Transformers."""
    if isinstance(saida_modelo, torch.Tensor):
        return saida_modelo
    if hasattr(saida_modelo, "pooler_output") and saida_modelo.pooler_output is not None:
        return saida_modelo.pooler_output
    if hasattr(saida_modelo, "last_hidden_state"):
        return saida_modelo.last_hidden_state[:, 0]
    return saida_modelo[0]


class ClassificadorMarcaModelo:
    """
    Classificador de Marca, Modelo e Carroceria (Fase 2 - VMMR).
    Por padrão utiliza o motor Zero-Shot Google SigLIP ('google/siglip-base-patch16-224')
    com Prompt Ensembling da frota brasileira e cache dos vetores de texto em RAM/disco.
    """

    def __init__(self):
        self.habilitado = getattr(config, "CAR_CLASSIFIER_ENABLED", True)
        self.engine = getattr(config, "CAR_CLASSIFIER_ENGINE", "siglip")
        self.device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"

        self.siglip_model = None
        self.siglip_processor = None
        self.nomes_modelos = []
        self.vetores_modelos = None     # Shape: [N_modelos, D]
        self.nomes_categorias = []
        self.vetores_categorias = None  # Shape: [N_categorias, D]

        self.modelo_yolo_cls = None

        if not self.habilitado:
            return

        # Opção A: Motor Zero-Shot Google SigLIP (Padrão)
        if self.engine == "siglip":
            if self._inicializar_siglip():
                return

        # Opção B: Modelo supervisionado YOLOv8-cls (.pt) caso configurado
        caminho_pt = getattr(
            config,
            "CAR_MODEL_WEIGHTS_PATH",
            os.path.join(config.BASE_DIR, "models", "car_yolov8_cls.pt"),
        )
        if os.path.exists(caminho_pt):
            try:
                from ultralytics import YOLO
                self.modelo_yolo_cls = YOLO(caminho_pt)
                print(f"[ClassificadorMarcaModelo] Modelo YOLO-cls carregado: {caminho_pt}")
            except Exception as e:
                print(f"[ClassificadorMarcaModelo] Aviso ao carregar YOLO-cls ({caminho_pt}): {e}")

    def _calcular_hash_catalogo(self, modelo_id: str) -> str:
        """Gera um hash único do catálogo de prompts + modelo para validar o cache em disco."""
        conteudo = f"{modelo_id}|{repr(CATALOGO_FROTA_BR)}|{repr(PROMPTS_CATEGORIAS)}"
        return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()[:16]

    def _inicializar_siglip(self) -> bool:
        """Carrega o modelo SigLIP e pré-calcula (ou restaura do cache) os vetores de texto."""
        modelo_id = getattr(config, "SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
        caminho_cache = getattr(
            config,
            "SIGLIP_CACHE_PATH",
            os.path.join(config.BASE_DIR, "models", "siglip_text_embeddings.pt"),
        )
        try:
            from transformers import AutoProcessor, AutoModel

            print(f"[ClassificadorMarcaModelo] Carregando SigLIP '{modelo_id}' em {self.device}...")
            try:
                self.siglip_processor = AutoProcessor.from_pretrained(modelo_id, local_files_only=True)
                self.siglip_model = AutoModel.from_pretrained(modelo_id, local_files_only=True).to(self.device).eval()
            except Exception:
                self.siglip_processor = AutoProcessor.from_pretrained(modelo_id)
                self.siglip_model = AutoModel.from_pretrained(modelo_id).to(self.device).eval()

            hash_atual = self._calcular_hash_catalogo(modelo_id)

            # 1. Tenta carregar os vetores de texto já pré-calculados do disco (instantâneo)
            if os.path.exists(caminho_cache):
                try:
                    dados_cache = torch.load(caminho_cache, map_location=self.device, weights_only=True)
                    if dados_cache.get("hash") == hash_atual:
                        self.nomes_modelos = dados_cache["nomes_modelos"]
                        self.vetores_modelos = dados_cache["vetores_modelos"].to(self.device)
                        self.nomes_categorias = dados_cache["nomes_categorias"]
                        self.vetores_categorias = dados_cache["vetores_categorias"].to(self.device)
                        print(
                            f"[ClassificadorMarcaModelo] Cache de texto SigLIP carregado "
                            f"({len(self.nomes_modelos)} modelos BR prontos na RAM)."
                        )
                        return True
                except Exception:
                    pass

            # 2. Pré-calcula os vetores de texto com Prompt Ensembling UMA ÚNICA VEZ
            print(f"[ClassificadorMarcaModelo] Pré-calculando embeddings de texto ({len(CATALOGO_FROTA_BR)} modelos BR)...")
            nomes_mod = []
            lista_vetores_mod = []

            with torch.no_grad():
                for nome_carro, info in CATALOGO_FROTA_BR.items():
                    prompts = info["prompts"]
                    inputs_txt = self.siglip_processor(
                        text=prompts, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(self.device)
                    emb = _extrair_tensor_embedding(self.siglip_model.get_text_features(**inputs_txt))
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    # Média dos prompts do mesmo modelo (Prompt Ensembling) e re-normalização
                    emb_medio = emb.mean(dim=0, keepdim=True)
                    emb_medio = emb_medio / emb_medio.norm(p=2, dim=-1, keepdim=True)
                    nomes_mod.append(nome_carro)
                    lista_vetores_mod.append(emb_medio)

                self.nomes_modelos = nomes_mod
                self.vetores_modelos = torch.cat(lista_vetores_mod, dim=0)

                nomes_cat = []
                lista_vetores_cat = []
                for nome_cat, prompts_cat in PROMPTS_CATEGORIAS.items():
                    inputs_cat = self.siglip_processor(
                        text=prompts_cat, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(self.device)
                    emb_c = _extrair_tensor_embedding(self.siglip_model.get_text_features(**inputs_cat))
                    emb_c = emb_c / emb_c.norm(p=2, dim=-1, keepdim=True)
                    emb_c_medio = emb_c.mean(dim=0, keepdim=True)
                    emb_c_medio = emb_c_medio / emb_c_medio.norm(p=2, dim=-1, keepdim=True)
                    nomes_cat.append(nome_cat)
                    lista_vetores_cat.append(emb_c_medio)

                self.nomes_categorias = nomes_cat
                self.vetores_categorias = torch.cat(lista_vetores_cat, dim=0)

            # Salva no disco para inicializações futuras instantâneas
            try:
                os.makedirs(os.path.dirname(caminho_cache), exist_ok=True)
                torch.save(
                    {
                        "hash": hash_atual,
                        "nomes_modelos": self.nomes_modelos,
                        "vetores_modelos": self.vetores_modelos.cpu(),
                        "nomes_categorias": self.nomes_categorias,
                        "vetores_categorias": self.vetores_categorias.cpu(),
                    },
                    caminho_cache,
                )
            except Exception:
                pass

            print(
                f"[ClassificadorMarcaModelo] SigLIP pronto! "
                f"{len(self.nomes_modelos)} modelos da frota BR cacheados em RAM."
            )
            return True

        except Exception as e:
            print(f"[ClassificadorMarcaModelo] Aviso ao inicializar SigLIP: {e}")
            return False

    def analisar_top_k(self, crop_veiculo: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """
        Retorna os Top-K modelos mais prováveis [(nome_modelo, probabilidade), ...]
        para inspeção e diagnóstico em cima de um recorte BGR (OpenCV) ou RGB.
        """
        if self.siglip_model is None or self.vetores_modelos is None:
            return []

        img_rgb = cv2.cvtColor(crop_veiculo, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        with torch.no_grad():
            inputs_img = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device)
            vetor_img = _extrair_tensor_embedding(self.siglip_model.get_image_features(**inputs_img))
            vetor_img = vetor_img / vetor_img.norm(p=2, dim=-1, keepdim=True)

            escala = self.siglip_model.logit_scale.exp()
            logits = vetor_img @ self.vetores_modelos.T * escala
            if hasattr(self.siglip_model, "logit_bias"):
                logits = logits + self.siglip_model.logit_bias
            probs = logits.softmax(dim=-1)[0]

            k_efetivo = min(k, len(self.nomes_modelos))
            top_probs, top_indices = torch.topk(probs, k_efetivo)

        return [
            (self.nomes_modelos[int(idx)], float(prob.item()))
            for prob, idx in zip(top_probs, top_indices)
        ]

    def calcular_confiabilidade(self, prob_1: float, prob_2: float = 0.0) -> str:
        """Calcula o nível na Matriz de Confiabilidade VMMR para o resultado."""
        return calcular_matriz_confiabilidade(prob_1, prob_2)

    def classificar(self, crop_veiculo: np.ndarray, tipo_veiculo: str = "Carro") -> dict:
        """
        Classifica a marca, modelo e categoria do veículo a partir de seu recorte (crop BGR).
        Retorna dicionário com:
            - marca_modelo: modelo vencedor
            - categoria: carroceria
            - confianca: score do 1º
            - top3: lista com os 3 mais prováveis e suas porcentagens formatadas
                    (ex: "1º Modelo (X%), 2º Modelo (Y%), 3º Modelo (Z%)")
            - confiabilidade: "Alta", "Média" ou "Baixa / Ambígua"
        """
        if not self.habilitado or crop_veiculo is None or crop_veiculo.size == 0:
            return {
                "marca_modelo": "-",
                "categoria": tipo_veiculo,
                "confianca": 0.0,
                "top3": "-",
                "confiabilidade": "Baixa / Ambígua",
            }

        h, w = crop_veiculo.shape[:2]
        if h < 40 or w < 40:
            return {
                "marca_modelo": "-",
                "categoria": tipo_veiculo,
                "confianca": 0.0,
                "top3": "-",
                "confiabilidade": "Baixa / Ambígua",
            }

        if tipo_veiculo != "Carro":
            return {
                "marca_modelo": tipo_veiculo,
                "categoria": tipo_veiculo,
                "confianca": 1.0,
                "top3": f"1º {tipo_veiculo} (100.0%)",
                "confiabilidade": "Alta",
            }

        # 1. Classificação Zero-Shot via Google SigLIP (Vision Encoder -> Produto Escalar com Cache RAM)
        if self.siglip_model is not None and self.vetores_modelos is not None:
            try:
                img_rgb = cv2.cvtColor(crop_veiculo, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                with torch.no_grad():
                    inputs_img = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device)
                    vetor_img = _extrair_tensor_embedding(self.siglip_model.get_image_features(**inputs_img))
                    vetor_img = vetor_img / vetor_img.norm(p=2, dim=-1, keepdim=True)

                    escala = self.siglip_model.logit_scale.exp()

                    # Similaridade contra os modelos específicos da frota BR
                    logits_mod = vetor_img @ self.vetores_modelos.T * escala
                    if hasattr(self.siglip_model, "logit_bias"):
                        logits_mod = logits_mod + self.siglip_model.logit_bias
                    probs_mod = logits_mod.softmax(dim=-1)[0]

                    k_efetivo = min(3, len(self.nomes_modelos))
                    top_probs, top_indices = torch.topk(probs_mod, k_efetivo)
                    top_probs_list = [float(p.item()) for p in top_probs]
                    top_nomes_list = [self.nomes_modelos[int(idx.item())] for idx in top_indices]

                    conf_mod = top_probs_list[0]
                    melhor_modelo = top_nomes_list[0]
                    prob_segundo = top_probs_list[1] if len(top_probs_list) > 1 else 0.0

                    # Formata Top-3 formatado: "1º Modelo (X%), 2º Modelo (Y%), 3º Modelo (Z%)"
                    top3_str = ", ".join(
                        f"{i+1}º {nome} ({p * 100:.1f}%)"
                        for i, (nome, p) in enumerate(zip(top_nomes_list, top_probs_list))
                    )

                    # Calcula Matriz de Confiabilidade
                    confiabilidade = self.calcular_confiabilidade(conf_mod, prob_segundo)

                    # Similaridade contra as categorias de carroceria (custo zero extra de imagem)
                    logits_cat = vetor_img @ self.vetores_categorias.T * escala
                    probs_cat = logits_cat.softmax(dim=-1)[0]
                    conf_cat_prob, top_cat_idx = torch.max(probs_cat, dim=0)
                    melhor_cat = self.nomes_categorias[int(top_cat_idx.item())]

                min_conf = getattr(config, "CAR_CLASSIFIER_MIN_CONF", 0.08)
                if conf_mod >= min_conf:
                    categoria_modelo = inferir_categoria_por_modelo(melhor_modelo)
                    return {
                        "marca_modelo": melhor_modelo,
                        "categoria": categoria_modelo,
                        "confianca": round(conf_mod, 3),
                        "top3": top3_str,
                        "confiabilidade": confiabilidade,
                    }
                else:
                    return {
                        "marca_modelo": melhor_cat,
                        "categoria": melhor_cat,
                        "confianca": round(float(conf_cat_prob.item()), 3),
                        "top3": top3_str,
                        "confiabilidade": "Baixa / Ambígua",
                    }
            except Exception as e:
                print(f"[ClassificadorMarcaModelo] Erro na inferência SigLIP: {e}")

        # 2. Alternativa: Inferência com YOLOv8-cls caso configurado
        if self.modelo_yolo_cls is not None:
            try:
                resultados = self.modelo_yolo_cls(crop_veiculo, verbose=False)
                if resultados and len(resultados) > 0:
                    res = resultados[0]
                    top1_idx = int(res.probs.top1)
                    nome_classe = str(res.names[top1_idx])
                    conf = float(res.probs.top1conf)

                    top5_indices = res.probs.top5 if hasattr(res.probs, "top5") else [top1_idx]
                    top_itens = []
                    for idx in top5_indices[:3]:
                        n = formatar_nome_veiculo(str(res.names[idx]))
                        p = float(res.probs.data[idx])
                        top_itens.append((n, p))
                    top3_str = ", ".join(f"{i+1}º {n} ({p*100:.1f}%)" for i, (n, p) in enumerate(top_itens))
                    prob_segundo = top_itens[1][1] if len(top_itens) > 1 else 0.0
                    confiabilidade = self.calcular_confiabilidade(conf, prob_segundo)

                    if conf >= getattr(config, "CAR_CLASSIFIER_MIN_CONF", 0.30):
                        marca_modelo_fmt = formatar_nome_veiculo(nome_classe)
                        categoria = inferir_categoria_por_modelo(marca_modelo_fmt)
                        return {
                            "marca_modelo": marca_modelo_fmt,
                            "categoria": categoria,
                            "confianca": round(conf, 2),
                            "top3": top3_str,
                            "confiabilidade": confiabilidade,
                        }
            except Exception:
                pass

        return {
            "marca_modelo": "Carro",
            "categoria": "Carro",
            "confianca": 0.0,
            "top3": "-",
            "confiabilidade": "Baixa / Ambígua",
        }


def classificar_veiculo(crop_veiculo: np.ndarray, cls_id: int = 2,
                        classificador_mm: ClassificadorMarcaModelo | None = None) -> dict:
    """
    Retorna dicionário consolidado com tipo, cor, e marca/modelo do veículo.
    """
    tipo = classificar_tipo_veiculo(cls_id)
    cor = estimar_cor_lataria(crop_veiculo)

    marca_modelo = "-"
    categoria = tipo
    confianca = 0.0
    top3 = "-"
    confiabilidade = "Baixa / Ambígua"

    if classificador_mm is not None:
        info_mm = classificador_mm.classificar(crop_veiculo, tipo)
        marca_modelo = info_mm.get("marca_modelo", "-")
        categoria = info_mm.get("categoria", tipo)
        confianca = info_mm.get("confianca", 0.0)
        top3 = info_mm.get("top3", "-")
        confiabilidade = info_mm.get("confiabilidade", "Baixa / Ambígua")

    return {
        "tipo": tipo,
        "cor": cor,
        "marca_modelo": marca_modelo,
        "categoria": categoria,
        "confianca": confianca,
        "top3": top3,
        "confiabilidade": confiabilidade,
    }
