# site/

The pages served at https://dmvjs.com/ket/ — hand-authored, not generated.

These used to live in `dist/`, which is gitignored and wiped by `npm run build`
(`rm -rf dist`). That destroyed every local copy, leaving S3 as the only one:
`docs.html` and `demo.html` sat at v0.6.1 for two weeks because no local file
existed to redeploy. They live here so a build can't eat them.

Deploy with `npm run deploy:site` — syncs this directory to
`s3://dmvjs.com/ket/` and invalidates the CloudFront distribution
(`E2NLJQLRPS1ZY5`, the one aliased to `dmvjs.com`; `www` doesn't serve `/ket/`).
The sync has no `--delete`, so it only ever adds or overwrites.
