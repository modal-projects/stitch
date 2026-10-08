"""Shell for task setup scripts that keeps a task repository to what the task starts from.

Images built by cloning a repository often keep its later history, the reference fix
included, where ``git log --all``, the reflog or ``git fsck`` finds it. SWE-bench Pro's
images keep it under remote refs and tags; the MiMo-V2.6 code images keep it in the
reflog and under upstream refs.
"""

# Run in the repository with HEAD at the task's start. It drops every ref HEAD does not
# contain, every remote, the stash and the reflog, and then the objects only they reached;
# refs to HEAD's own history (an old release tag) stay.
PRUNE_LATER_HISTORY = r"""git for-each-ref --format='%(if)%(symref)%(then)%(refname)%(end)' | sed '/^$/d' |
    while read -r ref; do git symbolic-ref --delete "$ref"; done
git for-each-ref --format='delete %(refname)' --no-merged=HEAD | git update-ref --stdin
git for-each-ref --format='delete %(refname)' refs/remotes | git update-ref --stdin
git remote | while read -r remote; do git config --remove-section "remote.$remote"; done
rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD
git stash clear
git reflog expire --expire=now --all
git gc --prune=now --quiet
"""
