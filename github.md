git branch --set-upstream-to=origin/master master
git push --set-upstream origin master

## throw some changes
git rev-parse --verify master >> .git/info/grafts
git filter-branch -- --all

