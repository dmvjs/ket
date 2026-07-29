#!/bin/sh
set -e
DIR="$(dirname "$0")"

# Copy the latest ket bundle into the guide's assets directory
cp "$DIR/../dist/ket.js" "$DIR/assets/ket.js"
echo "Copied dist/ket.js → assets/ket.js"

# Generate an inline blob-URL loader so each page embeds ket.js directly
# — zero HTTP round-trip, module starts parsing before OJS runtime initialises
KET_B64=$(python3 -c "import base64,sys; sys.stdout.write(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" "$DIR/assets/ket.js")

cat > "$DIR/assets/ket-head.html" << HTMLEOF
<script type="module">
(function(){
  var b=new Blob([atob("${KET_B64}")],{type:"application/javascript"});
  window.__ket_load=import(URL.createObjectURL(b));
})();
</script>
HTMLEOF

echo "Generated assets/ket-head.html ($(wc -c < "$DIR/assets/ket-head.html" | tr -d ' ') bytes inline)"
