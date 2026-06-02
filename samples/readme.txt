LocalAI Studio - sample files (DO NOT SHIP UNMODIFIED)
=======================================================

This folder contains example files that demonstrate how LocalAI Studio
can be customized for specific test environments. They are provided as
reference templates only — copy them out of this folder, edit them for
your own machines, and drop the edited copies into the live install
location described below.


WHAT IS IN HERE
---------------

  skus.json           An example SKU definition file. The included
                      examples happen to describe Windows 365 Cloud PC
                      configurations because that is what the author
                      uses to exercise the app across a range of
                      hardware tiers. Adapt the examples to whatever
                      machines you want LocalAI to recognise.

  Model-Guide.html    An example Model-Guide.html generated from the
                      author's own benchmark runs against the SKUs
                      defined in the accompanying skus.json. It is
                      provided so you can see what the page looks like
                      once you have actual benchmark results for your
                      machines. Re-generate your own version from your
                      own benchmark runs rather than relying on these
                      numbers.

  readme.txt          This file.


HOW TO USE skus.json
--------------------

  1. Copy samples\skus.json into the same folder as run.bat (the root
     of your LocalAI install).
  2. Edit it to describe the machines you actually want to test.
  3. Restart LocalAI Studio. The Benchmark tab will use your SKU
     definitions to pick sensible Quick / Extended default selections
     and to label runs in benchmark reports.

  Without a skus.json, LocalAI ships with a generic single-machine
  profile derived from the host it is running on.


HOW TO USE Model-Guide.html
---------------------------

  Copy samples\Model-Guide.html into the docs\ folder of your install,
  overwriting the version shipped with the release, if you want to
  preview the example layout populated with real benchmark data.

  The version in docs\Model-Guide.html that ships with LocalAI is the
  blank template that gets re-populated when you run benchmarks on
  your own machines. The sample here was rendered from the author's
  Windows 365 Cloud PC runs and is purely illustrative.


DISCLAIMERS
-----------

This is a personal hobby project by Ron Martinsen, shared for
entertainment and personal experimentation. It has no connection to,
endorsement by, or affiliation with his employer, his employer's
products, or any other organization. See DISCLAIMER.md at the repo
root for the full statement.

The examples in this folder are NOT:

  - An official Microsoft, Windows 365, Azure, or Microsoft Cloud
    artifact, deliverable, or supported integration.
  - A guarantee of any specific hardware allocation, performance
    characteristic, model support level, or SKU configuration on
    Windows 365 Cloud PCs or any other commercial product. The
    Windows 365 entries reflect what was provisioned for the author
    at a single point in time; published SKU specs, hardware mixes,
    GPU partitions, and capacity tiers change over time and may
    differ from what you see provisioned for your own tenant.
  - A statement of supported configurations, roadmap, or product
    direction by any vendor. Vendor support determinations always
    take precedence over anything observed here.
  - A reason to expect that a workload that runs against these
    examples will run, perform, or be supported on your own machines,
    your own Cloud PCs, or anyone else's hardware.

Provided AS-IS with no warranty of any kind. Edit, replace, or
delete the contents of this folder freely before relying on it for
anything.
